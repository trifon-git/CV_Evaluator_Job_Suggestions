import os
import sys
import importlib.util
import json
from dotenv import load_dotenv
import requests 
from pymongo import MongoClient
from langdetect import detect, LangDetectException
from datetime import datetime

print("--- exctract_store_skills.py: Script Started ---", flush=True)


HTML_CHUNK_SIZE_FOR_TEST = int(os.getenv('HTML_CHUNK_SIZE', 8000))
MAX_CHUNKS_FOR_TEST = int(os.getenv('MAX_CHUNKS_FOR_TEST', 10)) 
PRINT_PROCESSED_TEXT_IN_LOG = os.getenv('PRINT_PROCESSED_TEXT_IN_LOG', 'False').lower() == 'true'
MAX_SOURCE_TEXT_PRINT_SNIPPET_LENGTH = 5000


TIMESTAMP_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILENAME = f"test_llm_chunked_extractions_{TIMESTAMP_STR}.jsonl"
PROJECT_ROOT_FOR_OUTPUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE_PATH = os.path.join(PROJECT_ROOT_FOR_OUTPUT, "data", OUTPUT_FILENAME)
os.makedirs(os.path.join(PROJECT_ROOT_FOR_OUTPUT, "data"), exist_ok=True)
print(f"exctract_store_skills.py: Output (Chunked Method Only) will be saved to: {OUTPUT_FILE_PATH}", flush=True)


LANG_CODE_TO_NAME_MAP_TEST = {
    "en": "English", "da": "Danish", "de": "German", "sv": "Swedish",
    "no": "Norwegian", "es": "Spanish", "fr": "French", "nl": "Dutch",
}
EXPERIENCE_LEVEL_HIERARCHY_TEST = [
    "Entry-level", "Junior (0-2 years)", "Mid-level (3-5 years)", "Senior (5-10 years)",
    "Lead (7+ years, with leadership)", "Principal/Expert (10+ years, deep expertise)",
    "Manager", "Director", "Executive"
]

try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir)
    dotenv_path = os.path.join(project_root_dir, '.env')
    skill_extractor_module_name_on_disk = '[STEP3]llm_skill_extractor.py'
    skill_extractor_module_path = os.path.join(current_script_dir, skill_extractor_module_name_on_disk)
except Exception as e:
    print(f"exctract_store_skills.py: ERROR during path determination: {e}", flush=True)
    sys.exit(1)

print(f"exctract_store_skills.py: Current script directory: {current_script_dir}", flush=True)
print(f"exctract_store_skills.py: Determined project root: {project_root_dir}", flush=True)
print(f"exctract_store_skills.py: Expected .env path: {dotenv_path}", flush=True)
print(f"exctract_store_skills.py: Expected skill_extractor module path: {skill_extractor_module_path}", flush=True)
print("exctract_store_skills.py: Attempting to load .env file...", flush=True)

if os.path.exists(dotenv_path):
    loaded_dotenv = load_dotenv(dotenv_path=dotenv_path, override=True)
    if loaded_dotenv:
        print("exctract_store_skills.py: .env file loaded successfully.", flush=True)
    else:
        print("exctract_store_skills.py: WARNING: load_dotenv() returned False.", flush=True)
else:
    print(f"exctract_store_skills.py: WARNING: .env file NOT found at {dotenv_path}.", flush=True)

NGROK_API_URL_FROM_ENV = os.getenv('NGROK_API_URL')
OLLAMA_API_URL_FROM_ENV = os.getenv('OLLAMA_API_URL')
OLLAMA_MODEL_NAME_FROM_ENV = os.getenv('OLLAMA_MODEL_NAME')
MONGO_URI_FROM_ENV = os.getenv('MONGO_URI')
MONGO_DB_NAME_FROM_ENV = os.getenv('MONGO_DB_NAME')
MONGO_COLLECTION_FROM_ENV = os.getenv('MONGO_COLLECTION') # MONGO_COLLECTION_FROM_ENV = "backup_daily_20250521_022726"

if OLLAMA_API_URL_FROM_ENV and OLLAMA_MODEL_NAME_FROM_ENV:
    print(f"exctract_store_skills.py: [STEP3]llm_skill_extractor.py will use Local Ollama, Model: {OLLAMA_MODEL_NAME_FROM_ENV}", flush=True)
elif NGROK_API_URL_FROM_ENV:
    print(f"exctract_store_skills.py: [STEP3]llm_skill_extractor.py will use NGROK API: ", flush=True) # Consider adding the NGROK URL here if available
else:
    print("exctract_store_skills.py: WARNING: No LLM API URLs configured in .env for [STEP3]!", flush=True)



if not PRINT_PROCESSED_TEXT_IN_LOG:
    print(f"exctract_store_skills.py: MAX_SOURCE_TEXT_PRINT_SNIPPET_LENGTH (if not printing full to log): {MAX_SOURCE_TEXT_PRINT_SNIPPET_LENGTH}", flush=True)

print(f"exctract_store_skills.py: Attempting to import module: {skill_extractor_module_name_on_disk}", flush=True)
imported_llm_function = None

if not os.path.exists(skill_extractor_module_path):
    print(f"exctract_store_skills.py: ERROR: Module file not found at {skill_extractor_module_path}", flush=True)
    sys.exit(1)

try:
    module_name_for_import = 'skill_extractor_module_step3_for_test'
    spec = importlib.util.spec_from_file_location(module_name_for_import, skill_extractor_module_path)
    if spec is None:
        print(f"exctract_store_skills.py: ERROR: Could not load spec for module at {skill_extractor_module_path}", flush=True)
        sys.exit(1)
    skill_extractor_module_object = importlib.util.module_from_spec(spec)
    sys.modules[module_name_for_import] = skill_extractor_module_object
    spec.loader.exec_module(skill_extractor_module_object)
    imported_llm_function = getattr(skill_extractor_module_object, 'extract_job_details_with_llm', None)
    if imported_llm_function is None:
        imported_llm_function = getattr(skill_extractor_module_object, 'extract_skills_with_llm', None)
        if imported_llm_function:
            print("exctract_store_skills.py: WARNING: Using fallback function name 'extract_skills_with_llm'.")
        else:
            print("exctract_store_skills.py: ERROR: 'extract_job_details_with_llm' or 'extract_skills_with_llm' function not found.", flush=True)
            sys.exit(1)
    print(f"exctract_store_skills.py: Successfully imported LLM extraction function.", flush=True)
except Exception as e: print(f"exctract_store_skills.py: ERROR during module import: {e}", flush=True); import traceback; traceback.print_exc(); sys.exit(1)

# --- Helper Functions ---
def chunk_html_content(html_content, chunk_size):
    chunks = [];
    if not html_content or not isinstance(html_content, str): return chunks
    for i in range(0, len(html_content), chunk_size): chunks.append(html_content[i:i + chunk_size])
    return chunks

def save_extraction_to_file(data_to_save, source_details):
    """Appends a dictionary to the JSONL output file."""
    output_record = {
        "timestamp": datetime.now().isoformat(),
        "source_details": source_details,
        "extracted_data": data_to_save

    }
    try:
        with open(OUTPUT_FILE_PATH, 'a', encoding='utf-8') as f:
            json.dump(output_record, f, ensure_ascii=False)
            f.write('\n')
    except Exception as e:
        print(f"exctract_store_skills.py: ERROR writing to output file {OUTPUT_FILE_PATH}: {e}", flush=True)

# --- Main Test Execution Function (Chunked Only) ---
def test_llm_with_chunking(description_text, source_identifier, job_title_for_file="N/A", mongo_doc_id_for_file="N/A"):
    print(f"\n--- exctract_store_skills.py: Testing LLM WITH CHUNKING on text from: {source_identifier} ---", flush=True)
    if not (OLLAMA_API_URL_FROM_ENV or NGROK_API_URL_FROM_ENV): print("exctract_store_skills.py: ERROR: No API URL configured.", flush=True); return None
    if imported_llm_function is None: print("exctract_store_skills.py: ERROR: extraction function is not available.", flush=True); return None
    if not description_text or not isinstance(description_text, str) or not description_text.strip(): print("exctract_store_skills.py: INFO: Provided text is empty or invalid. Skipping.", flush=True); return None

    detected_main_language_name_test = "Unknown"
    try:
        sample_text_for_lang_detect_test = description_text[:min(len(description_text), 3000)]
        if sample_text_for_lang_detect_test.strip():
            lang_code_test = detect(sample_text_for_lang_detect_test)
            detected_main_language_name_test = LANG_CODE_TO_NAME_MAP_TEST.get(lang_code_test, lang_code_test.capitalize())

    except Exception: pass


    html_chunks_full = chunk_html_content(description_text, HTML_CHUNK_SIZE_FOR_TEST);
    if not html_chunks_full: print(f"exctract_store_skills.py: No chunks generated for {source_identifier}. Skipping.", flush=True); return None
    
    if len(html_chunks_full) >= MAX_CHUNKS_FOR_TEST:
        print(f"exctract_store_skills.py: SKIPPING {source_identifier}: Text split into {len(html_chunks_full)} chunks (>= {MAX_CHUNKS_FOR_TEST}). Not processing.", flush=True)
        return None 

    html_chunks_to_process = html_chunks_full
    print(f"exctract_store_skills.py: Processing {len(html_chunks_to_process)} chunk(s) for {source_identifier}.", flush=True)
    processed_text_concatenated_for_log = ""

    
    aggregated_details_accumulator_test = {
        "skills": set(),
        "experience_level_required": "Not specified",
        "language_requirements": set(),  # Changed to set for storing unique language names
        "education_level_preferred": "Not specified",
        "job_type": "Not specified"
    }
    total_api_calls = 0 # Removed llm_extracted_languages_map_test

    for i, chunk in enumerate(html_chunks_to_process):
        if PRINT_PROCESSED_TEXT_IN_LOG: processed_text_concatenated_for_log += chunk
        elif len(processed_text_concatenated_for_log) < MAX_SOURCE_TEXT_PRINT_SNIPPET_LENGTH : processed_text_concatenated_for_log += chunk

        print(f"  Processing chunk {i+1}/{len(html_chunks_to_process)} (length: {len(chunk)} chars) for {source_identifier}", flush=True)
        try:
            llm_response_dict = imported_llm_function(chunk); total_api_calls += 1
            if not llm_response_dict or not isinstance(llm_response_dict, dict): print(f"  WARNING for {source_identifier} chunk {i+1}: Invalid or no data dict from LLM. Data: {llm_response_dict}", flush=True); continue

            skills_from_chunk = llm_response_dict.get('skills', [])
            if isinstance(skills_from_chunk, list):
                for s in skills_from_chunk:
                    if isinstance(s, str) and s.strip():
                        aggregated_details_accumulator_test["skills"].add(s.strip())

            chunk_exp = llm_response_dict.get("experience_level_required"); current_agg_exp = aggregated_details_accumulator_test["experience_level_required"]
            if chunk_exp and chunk_exp != "Not specified":
                if current_agg_exp == "Not specified": aggregated_details_accumulator_test["experience_level_required"] = chunk_exp
                elif isinstance(chunk_exp, list): 
                    valid_chunk_exp = [ce for ce in chunk_exp if isinstance(ce, str) and ce.strip()]
                    if not valid_chunk_exp: # if chunk_exp is like ["", None]
                        pass # current_agg_exp remains unchanged
                    elif isinstance(current_agg_exp, list): 
                        current_agg_exp.extend(valid_chunk_exp)
                        aggregated_details_accumulator_test["experience_level_required"] = sorted(list(set(current_agg_exp)))
                    else: # current_agg_exp is a string
                        new_exp_list = [current_agg_exp] + valid_chunk_exp
                        aggregated_details_accumulator_test["experience_level_required"] = sorted(list(set(new_exp_list)))
                elif isinstance(current_agg_exp, list): # chunk_exp is a string
                     if chunk_exp not in current_agg_exp: 
                        current_agg_exp.append(chunk_exp)
                        aggregated_details_accumulator_test["experience_level_required"] = sorted(list(set(current_agg_exp)))
                elif isinstance(current_agg_exp, str) and current_agg_exp != chunk_exp : # both are strings
                    aggregated_details_accumulator_test["experience_level_required"] = sorted(list(set([current_agg_exp, chunk_exp])))
            
            for key_s in ["education_level_preferred", "job_type"]:
                chunk_val_s = llm_response_dict.get(key_s)
                if isinstance(chunk_val_s, str) and chunk_val_s.strip() and chunk_val_s != "Not specified": # Ensure it's a valid string
                    if aggregated_details_accumulator_test[key_s] == "Not specified": 
                        aggregated_details_accumulator_test[key_s] = chunk_val_s
                    # If already set, typically we prefer the first valid one found or a more specific logic if needed.
                    # For now, it keeps the first valid one.
            
            # Updated language requirements aggregation
            langs_from_chunk_llm = llm_response_dict.get('language_requirements', []) # Expected: list of strings
            if isinstance(langs_from_chunk_llm, list):
                for lang_name_str in langs_from_chunk_llm:
                    if isinstance(lang_name_str, str) and lang_name_str.strip():
                        aggregated_details_accumulator_test["language_requirements"].add(lang_name_str.strip())

        except Exception as e: print(f"  ERROR processing chunk {i+1} for {source_identifier} with LLM: {e}", flush=True); import traceback; traceback.print_exc()

    final_aggregated_dict_to_print = {}
    final_aggregated_dict_to_print['skills'] = sorted(list(aggregated_details_accumulator_test["skills"]))
    
    current_exp_values_test_agg = aggregated_details_accumulator_test["experience_level_required"]; chosen_experience_level_test_agg = "Not specified"
    if isinstance(current_exp_values_test_agg, list):
        valid_exp_levels_test_agg = [lvl for lvl in current_exp_values_test_agg if isinstance(lvl, str) and lvl.strip() and lvl != "Not specified"]
        if not valid_exp_levels_test_agg: chosen_experience_level_test_agg = "Not specified"
        elif len(valid_exp_levels_test_agg) == 1: chosen_experience_level_test_agg = valid_exp_levels_test_agg[0]
        else:
            highest_level_test_agg, highest_idx_test_agg = "Not specified", -1; temp_fallback_test = []
            for level in valid_exp_levels_test_agg:
                temp_fallback_test.append(level)
                try: idx_test = EXPERIENCE_LEVEL_HIERARCHY_TEST.index(level);
                except ValueError: idx_test = -1 
                if idx_test > highest_idx_test_agg: highest_idx_test_agg, highest_level_test_agg = idx_test, level
            chosen_experience_level_test_agg = highest_level_test_agg if highest_level_test_agg != "Not specified" else (temp_fallback_test[0] if temp_fallback_test else "Not specified")
    elif isinstance(current_exp_values_test_agg, str) and current_exp_values_test_agg.strip() and current_exp_values_test_agg != "Not specified": chosen_experience_level_test_agg = current_exp_values_test_agg
    final_aggregated_dict_to_print['experience_level_required'] = chosen_experience_level_test_agg
    
    final_aggregated_dict_to_print['education_level_preferred'] = aggregated_details_accumulator_test["education_level_preferred"]
    final_aggregated_dict_to_print['job_type'] = aggregated_details_accumulator_test["job_type"]

    # Final processing for language_requirements
    final_languages_set = aggregated_details_accumulator_test["language_requirements"].copy() # Work with a copy
    if detected_main_language_name_test != "Unknown":
        # Add detected main document language if not already present (case-insensitive check)
        already_present = False
        for lang_in_set in final_languages_set:
            if lang_in_set.lower() == detected_main_language_name_test.lower():
                already_present = True
                break
        if not already_present:
            final_languages_set.add(detected_main_language_name_test) # Add original casing

    final_aggregated_dict_to_print['language_requirements'] = sorted(list(final_languages_set))


    print(f"\n  exctract_store_skills.py: Final Aggregated Details for '{source_identifier}' (Chunked - {total_api_calls} API calls):", flush=True)
    print(json.dumps(final_aggregated_dict_to_print, indent=2, ensure_ascii=False), flush=True)

    save_extraction_to_file(
        final_aggregated_dict_to_print, 
        {"type": source_identifier, "method": "Chunked", "job_title": job_title_for_file, "mongo_id": str(mongo_doc_id_for_file)}
    )

    return final_aggregated_dict_to_print

    if PRINT_PROCESSED_TEXT_IN_LOG:
        print(f"    --- Source Text Processed (Chunked - {len(processed_text_concatenated_for_log)} chars) ---", flush=True)
        print(f"    '''\n{processed_text_concatenated_for_log.strip()}\n    '''", flush=True)
        print(f"    --- End Source Text for '{source_identifier}' (Chunked) ---", flush=True)


# --- Test Cases ---
sample_job_description_hardcoded = """
We are looking for a Senior Software Engineer with 5-7 years experience in Python, Django, and PostgreSQL.
The ideal candidate should also have strong problem-solving skills and be a great team player.
Fluency in English is required, Danish is a plus (Conversational). This is a Full-time position.
A Bachelor's degree in Computer Science or related field is preferred.
Familiarity with Docker and Kubernetes is a plus. You should be able to write clean code.
The job also requires good communication skills. We need someone who can handle complex tasks.
This role involves system architecture and design. Knowledge of cloud platforms like AWS or Azure is beneficial.
Experience with microservices and RESTful APIs is highly desired. Agile methodologies are a must.
"""
test_llm_with_chunking(sample_job_description_hardcoded, source_identifier="Hardcoded Sample")
# Removed direct test for hardcoded sample

if MONGO_URI_FROM_ENV and MONGO_DB_NAME_FROM_ENV and MONGO_COLLECTION_FROM_ENV:
    print("\n--- exctract_store_skills.py: Fetching samples from MongoDB ---", flush=True)
    mongo_client = None
    try:
        mongo_client = MongoClient(MONGO_URI_FROM_ENV, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command('ping'); print("exctract_store_skills.py: Successfully connected to MongoDB.", flush=True)
        db = mongo_client[MONGO_DB_NAME_FROM_ENV]; collection = db[MONGO_COLLECTION_FROM_ENV]
        mongo_samples = list(collection.find(
            {
                "html_content": {"$exists": True, "$ne": None, "$ne": ""},
                "Status": "active",
                "skills": {"$exists": False}  # Only check for missing skills
            },
            {"_id": 1, "Title": 1, "html_content": 1}
        ))
        if mongo_samples:
            print(f"exctract_store_skills.py: Found {len(mongo_samples)} active samples from MongoDB that are missing extraction data.", flush=True)
            for i, sample_doc in enumerate(mongo_samples):
                doc_id = sample_doc.get('_id') # Keep as ObjectId for query
                doc_id_str = str(doc_id)
                job_title = sample_doc.get("Title", f"Unknown Title {doc_id_str}")
                html_content = sample_doc.get("html_content")
                if not html_content or not html_content.strip():
                    print(f"exctract_store_skills.py: INFO: MongoDB Sample {i+1} - \"{job_title}\" has empty/whitespace html_content. Skipping.", flush=True); continue

                # Call the extraction function
                extracted_data = test_llm_with_chunking(
                    html_content,
                    source_identifier=f"MongoDB Sample {i+1}",
                    job_title_for_file=job_title,
                    mongo_doc_id_for_file=doc_id_str 
                )

                
                if extracted_data and isinstance(extracted_data, dict):
                    try:
                        
                        update_doc = {"$set": {}}
                       
                        # Ensure all expected fields are considered for update, even if empty lists (like skills or languages)
                        # or "Not specified" for string fields, if that's the desired behavior for overwriting.
                        # The current logic only sets fields if they have a "truthy" value or are non-empty lists.
                        # This should be fine for language_requirements as an empty list is a valid state.

                        for key, value in extracted_data.items():
                            # This existing logic should correctly handle language_requirements as a list of strings
                            if value is not None: # Allows empty lists to be set
                                if isinstance(value, list): # For skills, language_requirements
                                     update_doc["$set"][key] = value
                                elif isinstance(value, str) and value.strip(): # For exp, edu, job_type
                                     update_doc["$set"][key] = value
                                elif isinstance(value, str) and key in ["experience_level_required", "education_level_preferred", "job_type"] and value == "Not specified":
                                     update_doc["$set"][key] = value # Explicitly set "Not specified"

                        if update_doc["$set"]:
                            update_result = collection.update_one(
                                {"_id": doc_id},
                                update_doc
                            )

                            if update_result.matched_count > 0:
                                print(f"  exctract_store_skills.py: Successfully updated MongoDB document {doc_id_str} with extracted data.", flush=True)
                            else:
                                print(f"  exctract_store_skills.py: WARNING: MongoDB document {doc_id_str} not found for update after fetching.", flush=True)
                        else:
                            print(f"  exctract_store_skills.py: INFO: No valid data extracted for MongoDB document {doc_id_str}. Skipping update.", flush=True)


                    except Exception as e:
                        print(f"  exctract_store_skills.py: ERROR updating MongoDB document {doc_id_str}: {e}", flush=True)

                # Removed direct test for MongoDB samples
        else: print("exctract_store_skills.py: No suitable active samples found in MongoDB that are missing extraction data.", flush=True)
    except Exception as e: print(f"exctract_store_skills.py: ERROR connecting to or querying MongoDB: {e}", flush=True)
    finally:
        if mongo_client: mongo_client.close(); print("exctract_store_skills.py: MongoDB connection closed.", flush=True)
else: print("\n--- exctract_store_skills.py: Skipping MongoDB samples (config not fully set). ---", flush=True)


print("\n--- exctract_store_skills.py: Direct test of imported_llm_function with empty/None text ---", flush=True)
if imported_llm_function:
    try:
        print("exctract_store_skills.py: Testing with '   ' (whitespace only)", flush=True)
        empty_text_result = imported_llm_function("   "); print(f"exctract_store_skills.py: Result for '   ': {json.dumps(empty_text_result, indent=2, ensure_ascii=False)}", flush=True)
        print("exctract_store_skills.py: Testing with None", flush=True)
        none_text_result = imported_llm_function(None); print(f"exctract_store_skills.py: Result for None: {json.dumps(none_text_result, indent=2, ensure_ascii=False)}", flush=True)
    except Exception as e: print(f"exctract_store_skills.py: ERROR during empty/None text test: {e}", flush=True)
else: print("exctract_store_skills.py: Skipping direct empty/None text test (extractor not available).", flush=True)

print(f"\n--- exctract_store_skills.py: Script Finished. Output saved to {OUTPUT_FILE_PATH} ---", flush=True)
