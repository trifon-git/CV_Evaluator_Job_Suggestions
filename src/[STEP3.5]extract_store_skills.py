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

# Load environment variables
load_dotenv()

# MongoDB connection details
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('MONGO_DB_NAME')
COLLECTION_NAME = os.getenv('MONGO_COLLECTION')
HTML_CHUNK_SIZE = int(os.getenv('HTML_CHUNK_SIZE', 2000))
MAX_CHUNKS_PER_JOB = int(os.getenv('MAX_CHUNKS_PER_JOB', 10)) 

# Language mappings - we'll probably need to add more later
LANG_CODES = {
    "en": "English", "da": "Danish", "de": "German", "sv": "Swedish", 
    "no": "Norwegian", "es": "Spanish", "fr": "French", "nl": "Dutch",
    # TODO: Expand as needed
}

# Experience levels in order of hierarchy
EXP_LEVELS = [
    "Entry-level",
    "Junior (0-2 years)",
    "Mid-level (3-5 years)",
    "Senior (5-10 years)",
    "Lead (7+ years, with leadership)",
    "Principal/Expert (10+ years, deep expertise)",
    "Manager",
    "Director",
    "Executive"
]

def log(msg):
    # Simple timestamp + message logger
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def load_module_with_funky_name(file_path, module_name):
    # Need this because of the brackets in filenames
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Find and load our skill extractor module
skill_extractor_file = '[STEP3]llm_skill_extractor.py'
skill_extractor_path = os.path.join(os.path.dirname(__file__), skill_extractor_file)

if not os.path.exists(skill_extractor_path):
    log(f"ERROR: Can't find {skill_extractor_file} at {skill_extractor_path}")
    sys.exit(1)

# Load the module
extractor = load_module_with_funky_name(skill_extractor_path, 'skill_extractor_module_step3_for_3_5')

# Get the extraction function - try both possible names
extract_job_details = getattr(extractor, 'extract_job_details_with_llm', None)
if extract_job_details is None:
    extract_job_details = getattr(extractor, 'extract_skills_with_llm', None)
    if extract_job_details:
        log(f"WARNING: Using fallback function 'extract_skills_with_llm'. Maybe rename it to 'extract_job_details_with_llm'?")
    else:
        log(f"ERROR: No extraction function found in {skill_extractor_file}. Looking for 'extract_job_details_with_llm' or 'extract_skills_with_llm'.")
        sys.exit(1)


def chunk_html(html, size):
    # Split HTML into manageable chunks for processing
    if not html or not isinstance(html, str): 
        return []
    
    chunks = []
    for i in range(0, len(html), size):
        chunks.append(html[i:i + size])
    return chunks

def connect_mongo():
    # Set up MongoDB connection
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    return client, collection

def extract_and_store_job_details(job_id, html_content, collection):
    try:
        log(f"[Job: {job_id}] Starting extraction. HTML size: {len(html_content) if html_content else 0}")
        
        # Check if we have something to work with
        if not html_content or not html_content.strip(): 
            log(f"[Job: {job_id}] Empty HTML. Nothing to process.")
            collection.update_one({'_id': job_id}, {'$set': {'Extracted_Details_Status': 'Empty HTML', 'Skills': []}})
            return False

        # Try to detect the job posting language
        main_language = "Unknown"
        try:
            # Don't need to analyze the whole thing for language detection
            sample = html_content[:min(len(html_content), 3000)]
            if sample.strip(): 
                lang_code = detect(sample)
                main_language = LANG_CODES.get(lang_code, lang_code.capitalize())
                log(f"[Job: {job_id}] Language: {main_language} ({lang_code})")
            else:
                log(f"[Job: {job_id}] Can't detect language - sample text is empty.")
        except LangDetectException:
            log(f"[Job: {job_id}] Language detection failed.")
        except Exception as e:
            log(f"[Job: {job_id}] Language detection error: {str(e)}")

        # Break HTML into chunks for processing
        chunks = chunk_html(html_content, HTML_CHUNK_SIZE)
        if not chunks:
            log(f"[Job: {job_id}] Failed to create chunks. Skipping.")
            collection.update_one({'_id': job_id}, {'$set': {'Extracted_Details_Status': 'No Chunks Created', 'Skills': []}})
            return False

        # Don't process too many chunks
        if len(chunks) > MAX_CHUNKS_PER_JOB:
            log(f"[Job: {job_id}] Too many chunks ({len(chunks)}). Processing first {MAX_CHUNKS_PER_JOB}.")
            chunks = chunks[:MAX_CHUNKS_PER_JOB]
        
        # Setup accumulators for job details
        details = {
            "skills": set(),
            "experience_level_required": "Not specified", 
            "language_requirements": [], 
            "education_level_preferred": "Not specified",
            "job_type": "Not specified"
        }
        
        languages = {} # For deduping languages

        # Process each chunk and collect results
        for i, chunk in enumerate(chunks):
            # Extract details from this chunk
            chunk_data = extract_job_details(chunk)
            
            # Skip if we didn't get useful data
            if not chunk_data or not isinstance(chunk_data, dict):
                continue

            # Collect skills
            chunk_skills = chunk_data.get('skills', [])
            if isinstance(chunk_skills, list):
                for skill in chunk_skills:
                    if isinstance(skill, str) and skill.strip():
                         details["skills"].add(skill.strip())
            
            # Handle experience level - this is more complex because multiple values could conflict
            chunk_exp = chunk_data.get("experience_level_required")
            current_exp = details["experience_level_required"]
            
            if chunk_exp and chunk_exp != "Not specified":
                # Various cases for current and new experience data
                if current_exp == "Not specified":
                    # Easy case - just use the new value
                    details["experience_level_required"] = chunk_exp
                elif isinstance(chunk_exp, list): 
                    # New value is a list
                    if isinstance(current_exp, list): 
                        # Both are lists - combine and dedupe
                        details["experience_level_required"] = sorted(list(set(current_exp + chunk_exp)))
                    else: 
                        # Current is string, new is list - combine and dedupe
                        details["experience_level_required"] = sorted(list(set([current_exp] + chunk_exp)))
                elif isinstance(current_exp, list): 
                    # Current is list, new is string
                    if chunk_exp not in current_exp:
                        current_exp.append(chunk_exp)
                        details["experience_level_required"] = sorted(list(set(current_exp)))
                elif current_exp != chunk_exp: 
                    # Both are strings but different - make a list
                    details["experience_level_required"] = sorted(list(set([current_exp, chunk_exp])))

            # Handle simpler fields - take first non-default value
            for key in ["education_level_preferred", "job_type"]:
                chunk_value = chunk_data.get(key)
                if chunk_value and chunk_value != "Not specified":
                    if details[key] == "Not specified":
                         details[key] = chunk_value
            
            # Process language requirements
            chunk_langs = chunk_data.get('language_requirements', [])
            if isinstance(chunk_langs, list):
                for lang_obj in chunk_langs:
                    if isinstance(lang_obj, dict) and "language" in lang_obj and isinstance(lang_obj["language"], str):
                        lang_name = lang_obj["language"].strip()
                        lang_key = lang_name.lower()  # For case-insensitive deduping
                        proficiency = lang_obj.get("proficiency", "Not specified").strip()
                        
                        if lang_name:
                            # Only override if new proficiency is better than existing
                            if lang_key not in languages or \
                               (languages[lang_key]["proficiency"] == "Not specified" and proficiency != "Not specified"):
                                languages[lang_key] = {"language": lang_name, "proficiency": proficiency}
        
        # Prepare the MongoDB update
        update = {}
        update['Skills'] = sorted(list(details["skills"]))
        
        # Process experience levels to pick the most relevant one
        exp_values = details["experience_level_required"]
        chosen_exp = "Not specified"

        if isinstance(exp_values, list):
            # Filter out invalid values
            valid_levels = [
                level for level in exp_values 
                if isinstance(level, str) and level.strip() and level != "Not specified"
            ]
            
            if not valid_levels:
                chosen_exp = "Not specified"
            elif len(valid_levels) == 1:
                chosen_exp = valid_levels[0]
            else:
                # Multiple levels - try to find the highest in our hierarchy
                highest_level = "Not specified"
                highest_index = -1
                fallback_levels = []
                
                for level in valid_levels:
                    fallback_levels.append(level)
                    try:
                        # Check if this level is in our hierarchy
                        idx = EXP_LEVELS.index(level)
                        if idx > highest_index:
                            highest_index = idx
                            highest_level = level
                    except ValueError:
                        log(f"[Job: {job_id}] Non-standard experience level: '{level}'")
                
                if highest_level != "Not specified":
                     chosen_exp = highest_level
                elif fallback_levels: 
                     # If no standard levels found, use the first non-standard one
                     chosen_exp = fallback_levels[0]
                else:
                     chosen_exp = "Not specified"

        elif isinstance(exp_values, str) and exp_values.strip() and exp_values != "Not specified":
            chosen_exp = exp_values
        
        update['Experience_Level_Required'] = chosen_exp

        # Add other fields if we have values
        if details["education_level_preferred"] != "Not specified":
            update['Education_Level_Preferred'] = details["education_level_preferred"]
        if details["job_type"] != "Not specified":
            update['Job_Type'] = details["job_type"]
        
        # Process language requirements
        final_languages = []
        processed_langs = set()

        # Add languages from extraction
        for lang_key, lang_data in languages.items():
            final_languages.append(lang_data)
            processed_langs.add(lang_key)
        
        # Add detected document language if not already included
        if main_language != "Unknown":
            main_lang_key = main_language.lower()
            if main_lang_key not in processed_langs:
                # Add primary language with "Fluent" proficiency
                final_languages.append({"language": main_language, "proficiency": "Fluent"}) 
                processed_langs.add(main_lang_key)
            else: 
                # Update proficiency of already-found language if needed
                for lang_entry in final_languages:
                    if lang_entry["language"].lower() == main_lang_key and lang_entry["proficiency"] == "Not specified":
                        lang_entry["proficiency"] = "Fluent" 
                        break
        
        if final_languages:
            update['Language_Requirements'] = sorted(final_languages, key=lambda x: x['language'])
        
        # Set status fields
        update['Extracted_Details_Status'] = 'Success'
        update['Details_Extracted_At'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check if we actually found anything useful
        has_data = False
        if update.get('Skills'): 
            has_data = True
        if update.get('Experience_Level_Required', "Not specified") != "Not specified": 
            has_data = True
        if update.get('Education_Level_Preferred', "Not specified") != "Not specified": 
            has_data = True
        if update.get('Job_Type', "Not specified") != "Not specified": 
            has_data = True
        if update.get('Language_Requirements'): 
            has_data = True
        
        # Update the database
        if has_data:
            collection.update_one(
                {'_id': job_id},
                {'$set': update}
            )
            # Pretty-print the interesting parts of the update
            log_data = {k:v for k,v in update.items() if k not in ['Extracted_Details_Status', 'Details_Extracted_At']}
            log(f"[Job: {job_id}] Success! Found: {json.dumps(log_data, indent=2, ensure_ascii=False)}")
            return True
        else:
            log(f"[Job: {job_id}] No useful data extracted.")
            collection.update_one(
                {'_id': job_id}, 
                {'$set': {'Extracted_Details_Status': 'No New Details Extracted', 'Skills': update.get('Skills',[])}}
            )
            return False
            
    except Exception as e:
        log(f"[Job: {job_id}] CRITICAL ERROR: {str(e)}")
        traceback.print_exc()
        try:
            collection.update_one(
                {'_id': job_id}, 
                {'$set': {'Extracted_Details_Status': f'Error: {str(e)}', 'Skills': []}}
            )
        except Exception as db_err:
            log(f"[Job: {job_id}] Couldn't even update error status: {db_err}")
        return False

def process_jobs():
    try:
        client, collection = connect_mongo()
        
        # Find jobs that need processing
        query = {
            'html_content': {'$exists': True, '$ne': None, '$ne': ""},
            'Status': 'active', 
            '$or': [
                {'Details_Extracted_At': {'$exists': False}},
                {'Extracted_Details_Status': {'$exists': False}},
                {'Extracted_Details_Status': {'$regex': '^Error'}}, 
                {'Extracted_Details_Status': 'Empty HTML'}, 
                {'Extracted_Details_Status': 'No Chunks Created'}, 
            ]
        }
        
        log(f"Looking for jobs to process: {json.dumps(query)}")
        
        # Count matching jobs
        try: 
            job_count = collection.count_documents(query)
        except Exception: 
            # Fallback if count_documents fails
            job_count = len(list(collection.find(query, {"_id": 1}))) 
            
        log(f"Found {job_count} jobs to process")
        
        if job_count == 0:
            log("Nothing to do. Exiting.")
            return
        
        # Process the jobs
        jobs = collection.find(query) 
        processed = 0
        successful = 0
        progress = tqdm(total=job_count, desc="Processing Jobs", unit="job")
        
        for job in jobs:
            job_id = job['_id']
            html = job.get('html_content')
            
            if not html or not html.strip():
                log(f"[Job: {job_id}] No HTML content despite query filter. Weird.")
                collection.update_one(
                    {'_id': job_id}, 
                    {'$set': {'Extracted_Details_Status': 'Missing/Empty HTML (Skipped)'}}
                )
                processed += 1
                progress.update(1)
                continue
            
            # Process this job
            if extract_and_store_job_details(job_id, html, collection):
                successful += 1
                
            processed += 1
            progress.update(1)
        
        progress.close()
        log(f"Finished! Processed {processed} jobs, successfully extracted {successful}")
        
    except Exception as e:
        log(f"Error in main process: {e}")
        traceback.print_exc()
    finally:
        if 'client' in locals() and client: 
            client.close()
            log("MongoDB connection closed")

if __name__ == "__main__":
    log("Starting job extraction process...")
    process_jobs()
