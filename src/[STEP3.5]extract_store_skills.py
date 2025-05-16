import os
import sys
import importlib.util
from datetime import datetime
from pymongo import MongoClient
from tqdm import tqdm
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
import json
import traceback

# Environment setup
load_dotenv()

# DB config
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('MONGO_DB_NAME')
COLLECTION_NAME = os.getenv('MONGO_COLLECTION') # Main jobs collection
HTML_CHUNK_SIZE = int(os.getenv('HTML_CHUNK_SIZE', 2000))
MAX_CHUNKS_PER_JOB = int(os.getenv('MAX_CHUNKS_PER_JOB', 10))

# Available language mappings
LANG_CODES = {
    "en": "English", "da": "Danish", "de": "German", "sv": "Swedish",
    "no": "Norwegian", "es": "Spanish", "fr": "French", "nl": "Dutch",
    # Add more as needed when we encounter them
}

# Experience levels from most to least senior
EXPERIENCE_LEVEL_HIERARCHY = [
    "Executive",
    "Director",
    "Manager",
    "Principal/Expert (10+ years, deep expertise)",
    "Lead (7+ years, with leadership)",
    "Senior (5-10 years)",
    "Mid-level (3-5 years)",
    "Junior (0-2 years)",
    "Entry-level"
]

def log(msg):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def load_llm_extraction_function():
    skill_extractor_file = '[STEP3]llm_skill_extractor.py'
    skill_extractor_path = os.path.join(os.path.dirname(__file__), skill_extractor_file)

    if not os.path.exists(skill_extractor_path):
        log(f"CRITICAL ERROR: LLM extractor file '{skill_extractor_file}' not found at '{skill_extractor_path}'")
        sys.exit(1)

    try:
        # Unique module name to avoid namespace conflicts
        module_name = f"llm_extractor_module_{datetime.now().timestamp()}" 
        spec = importlib.util.spec_from_file_location(module_name, skill_extractor_path)
        if spec is None:
            log(f"CRITICAL ERROR: Could not create spec for module {skill_extractor_file}")
            sys.exit(1)
        
        extractor_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = extractor_module # Needed for proper module importing
        spec.loader.exec_module(extractor_module)
        
        # Try main function name, then legacy name as fallback
        llm_function = getattr(extractor_module, 'extract_job_details_with_llm', None)
        if llm_function is None:
            llm_function = getattr(extractor_module, 'extract_skills_with_llm', None)
            if llm_function:
                log("WARNING: Using fallback LLM function name 'extract_skills_with_llm'. "
                    "Consider renaming to 'extract_job_details_with_llm' in [STEP3]llm_skill_extractor.py for clarity.")
            else:
                log("CRITICAL ERROR: Neither 'extract_job_details_with_llm' nor 'extract_skills_with_llm' "
                    f"function found in {skill_extractor_file}.")
                sys.exit(1)
        return llm_function
    except Exception as e:
        log(f"CRITICAL ERROR: Failed to load LLM extraction function from {skill_extractor_file}: {e}")
        traceback.print_exc()
        sys.exit(1)

# Load extraction function at script startup
extract_job_details_from_llm = load_llm_extraction_function()

def chunk_html_text(html_text, chunk_size):
    if not html_text or not isinstance(html_text, str):
        return []
    return [html_text[i:i + chunk_size] for i in range(0, len(html_text), chunk_size)]

def connect_to_mongodb():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000) # 5s timeout
        client.admin.command('ping') # Connection check
        collection = client[DB_NAME][COLLECTION_NAME]
        log("Successfully connected to MongoDB.")
        return client, collection
    except Exception as e:
        log(f"CRITICAL ERROR: Failed to connect to MongoDB at {MONGO_URI}. Error: {e}")
        sys.exit(1) # Fatal error - can't proceed without DB

def determine_final_experience(extracted_levels_list, job_id_for_log):
    """Picks most senior experience level from possible matches."""
    if not extracted_levels_list:
        return "Not specified"

    # Handle nested lists and filter out empties
    flat_levels = []
    for item in extracted_levels_list:
        if isinstance(item, list):
            flat_levels.extend(lvl_str for lvl_str in item if isinstance(lvl_str, str) and lvl_str.strip() and lvl_str != "Not specified")
        elif isinstance(item, str) and item.strip() and item != "Not specified":
            flat_levels.append(item)
    
    if not flat_levels:
        return "Not specified"

    unique_valid_levels = sorted(list(set(flat_levels)))

    best_hierarchical_level = "Not specified"
    highest_found_index = -1 

    for level_from_llm in unique_valid_levels:
        # Try exact matches first
        if level_from_llm in EXPERIENCE_LEVEL_HIERARCHY:
            try:
                current_index = EXPERIENCE_LEVEL_HIERARCHY.index(level_from_llm)
                if current_index > highest_found_index: # Lower index = more senior in our list
                    highest_found_index = current_index
                    best_hierarchical_level = level_from_llm
            except ValueError:
                pass 
        else:
            # Try fuzzy matching with keywords
            found_match_by_keyword = False
            for i, predefined_level in enumerate(EXPERIENCE_LEVEL_HIERARCHY):
                predefined_keyword = predefined_level.split(" (")[0] # Extract base title
                if predefined_keyword.lower() in level_from_llm.lower():
                    if i > highest_found_index:
                        highest_found_index = i
                        best_hierarchical_level = predefined_level # Use our standard format
                    found_match_by_keyword = True
                    break
            if not found_match_by_keyword:
                 log(f"[Job: {job_id_for_log}] Experience level '{level_from_llm}' from LLM not in hierarchy and no keyword match.")


    if best_hierarchical_level != "Not specified":
        return best_hierarchical_level
    elif unique_valid_levels: # Use first valid if no match in hierarchy
        log(f"[Job: {job_id_for_log}] No hierarchical match for experience: {unique_valid_levels}. Taking first valid: '{unique_valid_levels[0]}'")
        return unique_valid_levels[0]

    return "Not specified"


def extract_and_save_details_for_job(job_id, html_content, mongo_collection_obj):
    try:
        
        if not html_content or not html_content.strip():
            log(f"[Job: {job_id}] HTML content is empty. Marking and skipping.")
            mongo_collection_obj.update_one({'_id': job_id}, {'$set': {'Extracted_Details_Status': 'Empty HTML', 'Skills': []}})
            return False

        detected_doc_language = "Unknown"
        try:
            sample_text = html_content[:min(len(html_content), 3000)] # Larger sample improves detection accuracy
            if sample_text.strip():
                lang_code = detect(sample_text)
                detected_doc_language = LANG_CODES.get(lang_code, lang_code.capitalize())
        except Exception:
            log(f"[Job: {job_id}] Warning: Language detection failed for the document.")

        chunks = chunk_html_text(html_content, HTML_CHUNK_SIZE)
        if not chunks:
            log(f"[Job: {job_id}] No chunks created from HTML. Marking and skipping.")
            mongo_collection_obj.update_one({'_id': job_id}, {'$set': {'Extracted_Details_Status': 'No Chunks Created', 'Skills': []}})
            return False

        if len(chunks) > MAX_CHUNKS_PER_JOB:
            log(f"[Job: {job_id}] WARNING: Exceeded MAX_CHUNKS_PER_JOB ({MAX_CHUNKS_PER_JOB}). Processing first {MAX_CHUNKS_PER_JOB} chunks.")
            chunks = chunks[:MAX_CHUNKS_PER_JOB]
        
        # Result accumulators
        all_skills_found = set()
        all_experience_levels_found_raw = [] # Raw experience levels before resolution
        final_education = "Not specified"
        final_job_type = "Not specified"
        # Track languages by lowercase name -> {original name, best proficiency}
        aggregated_languages = {}

        for i, chunk_text in enumerate(chunks):
           
            llm_extracted_data = extract_job_details_from_llm(chunk_text) 
            
            if not llm_extracted_data or not isinstance(llm_extracted_data, dict):
                continue

            # Collect skills
            chunk_skills = llm_extracted_data.get('skills', [])
            if isinstance(chunk_skills, list):
                for skill in chunk_skills:
                    if isinstance(skill, str) and skill.strip():
                        all_skills_found.add(skill.strip())
            
            # Collect experience levels
            chunk_exp = llm_extracted_data.get("experience_level_required")
            if chunk_exp and chunk_exp != "Not specified":
                if isinstance(chunk_exp, list):
                    all_experience_levels_found_raw.extend(
                        item for item in chunk_exp if isinstance(item, str) and item.strip() and item != "Not specified")
                elif isinstance(chunk_exp, str) and chunk_exp.strip():
                    all_experience_levels_found_raw.append(chunk_exp)

            # Take first non-default education level
            if final_education == "Not specified":
                edu_val = llm_extracted_data.get("education_level_preferred")
                if edu_val and edu_val != "Not specified": final_education = edu_val
            
            # Take first non-default job type
            if final_job_type == "Not specified":
                job_type_val = llm_extracted_data.get("job_type")
                if job_type_val and job_type_val != "Not specified": final_job_type = job_type_val

            # Process language requirements
            chunk_langs = llm_extracted_data.get('language_requirements', [])
            if isinstance(chunk_langs, list):
                for lang_obj in chunk_langs:
                    if isinstance(lang_obj, dict) and "language" in lang_obj and isinstance(lang_obj["language"], str):
                        lang_name_orig = lang_obj["language"].strip()
                        lang_name_lower = lang_name_orig.lower()
                        proficiency = lang_obj.get("proficiency", "Not specified").strip()
                        if lang_name_orig:
                            # Keep best proficiency info if we see this language again
                            if lang_name_lower not in aggregated_languages or \
                               (aggregated_languages[lang_name_lower]["proficiency"] == "Not specified" and proficiency != "Not specified"):
                                aggregated_languages[lang_name_lower] = {"language": lang_name_orig, "proficiency": proficiency}
        
        # Prepare MongoDB update
        db_update_doc = {}
        db_update_doc['Skills'] = sorted(list(all_skills_found))
        
        # Apply experience level hierarchy rules
        db_update_doc['Experience_Level_Required'] = determine_final_experience(all_experience_levels_found_raw, job_id)
        
        if final_education != "Not specified": db_update_doc['Education_Level_Preferred'] = final_education
        if final_job_type != "Not specified": db_update_doc['Job_Type'] = final_job_type
        
        # Handle document language - assume fluency in ad language if not specified otherwise
        final_lang_list_for_db = list(aggregated_languages.values())
        if detected_doc_language != "Unknown":
            doc_lang_lower = detected_doc_language.lower()
            found_doc_lang_in_llm_output = False
            for lang_entry in final_lang_list_for_db:
                if lang_entry["language"].lower() == doc_lang_lower:
                    found_doc_lang_in_llm_output = True
                    # Default to fluent if proficiency not specified for doc language
                    if lang_entry.get("proficiency", "Not specified") == "Not specified":
                        lang_entry["proficiency"] = "Fluent" 
                    break
            if not found_doc_lang_in_llm_output:
                final_lang_list_for_db.append({"language": detected_doc_language, "proficiency": "Fluent"})
        
        if final_lang_list_for_db:
            db_update_doc['Language_Requirements'] = sorted(final_lang_list_for_db, key=lambda x: x['language'])

        db_update_doc['Extracted_Details_Status'] = 'Success'
        db_update_doc['Details_Extracted_At'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db_update_doc['Detected_Ad_Language'] = detected_doc_language

        # Check if we found anything useful
        has_meaningful_data = bool(db_update_doc.get('Skills')) or \
                              (db_update_doc.get('Experience_Level_Required', "Not specified") != "Not specified") or \
                              (db_update_doc.get('Education_Level_Preferred', "Not specified") != "Not specified") or \
                              (db_update_doc.get('Job_Type', "Not specified") != "Not specified") or \
                              bool(db_update_doc.get('Language_Requirements'))
        
        if has_meaningful_data:
            mongo_collection_obj.update_one({'_id': job_id}, {'$set': db_update_doc})
            log(f"[Job: {job_id}] Successfully extracted and stored details.")
            return True
        else:
            log(f"[Job: {job_id}] No substantive details extracted after processing. Marking.")
            mongo_collection_obj.update_one(
                {'_id': job_id}, 
                {'$set': {'Extracted_Details_Status': 'No New Details Extracted', 
                          'Skills': db_update_doc.get('Skills',[]), # Save skills even if nothing else found
                          'Details_Extracted_At': db_update_doc['Details_Extracted_At']
                         }}
            )
            return False
            
    except Exception as e:
        log(f"[Job: {job_id}] CRITICAL ERROR during extraction/storage: {str(e)}")
        traceback.print_exc()
        try:
            mongo_collection_obj.update_one({'_id': job_id}, {'$set': {'Extracted_Details_Status': f'Error: {str(e)}', 'Skills': []}})
        except Exception as db_err:
            log(f"[Job: {job_id}] FAILED to update error status in DB: {db_err}")
        return False

def main_process():
    log("Starting job detail extraction process...")
    client, mongo_collection = None, None # Init for finally block
    try:
        client, mongo_collection = connect_to_mongodb()
        
        query = {
            'html_content': {'$exists': True, '$ne': None, '$ne': ""},
            'Status': 'active', 
            '$or': [
                {'Details_Extracted_At': {'$exists': False}},
                {'Extracted_Details_Status': {'$exists': False}},
                {'Extracted_Details_Status': {'$regex': '^Error', '$options': 'i'}}, 
                {'Extracted_Details_Status': 'Empty HTML'}, 
                {'Extracted_Details_Status': 'No Chunks Created'},
                {'Extracted_Details_Status': 'No New Details Extracted'} 
            ]
        }
        
        job_count_to_process = mongo_collection.count_documents(query)
        log(f"Found {job_count_to_process} jobs to process for detail extraction.")
        
        if job_count_to_process == 0:
            log("No jobs to process. Exiting.")
            return
        
        jobs_to_process_cursor = mongo_collection.find(query)
        
        processed_job_count = 0
        successful_job_count = 0
        progress_bar = tqdm(total=job_count_to_process, desc="Extracting Job Details", unit="job")
        
        for job_document in jobs_to_process_cursor:
            job_id = job_document['_id']
            html_content_from_db = job_document.get('html_content')
            
            if not html_content_from_db or not html_content_from_db.strip():
                log(f"[Job: {job_id}] Skipped: html_content field is missing or empty in DB doc (unexpected).")
                mongo_collection.update_one({'_id': job_id}, {'$set': {'Extracted_Details_Status': 'Missing/Empty HTML (Skipped)'}})
                processed_job_count += 1
                progress_bar.update(1)
                continue
            
            if extract_and_save_details_for_job(job_id, html_content_from_db, mongo_collection):
                successful_job_count += 1
            
            processed_job_count += 1
            progress_bar.update(1)
        
        progress_bar.close()
        log(f"Processing complete. Processed: {processed_job_count}, Successfully extracted/updated details for: {successful_job_count} jobs.")
        
    except Exception as e:
        log(f"FATAL ERROR in main_process: {e}")
        traceback.print_exc()
    finally:
        if client:
            client.close()
            log("MongoDB connection closed.")

if __name__ == "__main__":
    main_process()
