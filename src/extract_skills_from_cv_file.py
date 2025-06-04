import os
import requests
import json
import re
from dotenv import load_dotenv

# --- Configuration ---
SYSTEM_PROMPT = """ 
 You are a Head HR Manager with 20 years of experience in talent acquisition and skills assessment both in English and in Danish. 
 
 Your TASK: 
 Extract ALL skills mentioned in the job description text and return ONLY a list of skills in JSON format. 
 
 Output format (MUST be exactly this format): 
 {"skills": ["Skill 1", "Skill 2", "Skill 3", ...]} 
 
 IMPORTANT RULES: 
 1. DO NOT return Schema.org metadata or any webpage information 
 2. DO NOT return HTML or structured data about the webpage 
 3. ONLY extract skills mentioned in the text 
 4. Your ENTIRE response must be ONLY the JSON object with skills 
 
 Guidelines for skill extraction: 
 - Include both technical skills and soft skills 
 - Include both specific skills and general ones 
 - Do not include job titles, degrees, or certifications 
 - Return an empty list if no skills are found: {"skills": []} 
 """

CV_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "data", 
    "cv_text_example.md"
)

# Chunking Configuration
MAX_CHUNK_CHAR_LENGTH = 2000  # Max characters per chunk (adjust based on model context window and typical CV length)
CHUNK_OVERLAP_CHAR_LENGTH = 300   # Characters to overlap between chunks

# --- Load Environment Variables ---
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir)
    dotenv_path = os.path.join(project_root_dir, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path, override=True)
        print(f"INFO: .env file loaded successfully")
    else:
        print(f"WARNING: .env file NOT found. API calls might fail.")
except Exception as e:
    print(f"ERROR during .env loading: {e}")

CUSTOM_LLM_API_URL = os.getenv("CUSTOM_LLM_API_URL")
CUSTOM_LLM_MODEL_NAME = os.getenv("CUSTOM_LLM_MODEL_NAME")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL_NAME")
NGROK_API_URL = os.getenv("NGROK_API_URL")

API_URL = None
MODEL_TO_USE = None
API_TYPE = "unconfigured" # Possible values: 'custom', 'ollama', 'ngrok', 'unconfigured'

if CUSTOM_LLM_API_URL and CUSTOM_LLM_MODEL_NAME:
    API_URL = CUSTOM_LLM_API_URL
    MODEL_TO_USE = CUSTOM_LLM_MODEL_NAME
    API_TYPE = "custom"
    print(f"INFO: Using Custom LLM API with model: {MODEL_TO_USE}")
elif OLLAMA_API_URL and OLLAMA_MODEL:
    API_URL = OLLAMA_API_URL
    MODEL_TO_USE = OLLAMA_MODEL
    API_TYPE = "ollama"
    print(f"INFO: Using Local Ollama API with model: {MODEL_TO_USE}")
elif NGROK_API_URL:
    API_URL = NGROK_API_URL
    MODEL_TO_USE = OLLAMA_MODEL
    API_TYPE = "ngrok"
    print(f"INFO: Using NGROK (hosted) API with model: {MODEL_TO_USE if MODEL_TO_USE else 'default'}")
else:
    print("ERROR: No suitable API URL (Custom, Ollama, or NGROK) is set in .env. API calls will likely fail.")
    # API_URL remains None, API_TYPE remains 'unconfigured'

def create_text_chunks(text, max_length, overlap):
    """
    Splits a text into overlapping chunks.
    """
    if not text or not isinstance(text, str):
        return []
        
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_length, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start += max_length - overlap
        if start >= text_len: # Ensure we don't create an empty or tiny chunk at the very end due to overlap
            break
        # Ensure the next chunk doesn't start beyond the text length if overlap is large
        start = min(start, text_len -1) if text_len > 0 else 0


    # Filter out any potentially empty strings if text was very short or only whitespace
    return [chunk for chunk in chunks if chunk.strip()]


def extract_skills_from_text(text_chunk):
    """
    Extracts skills from the given text CHUNK using an LLM.
    """
    if not API_URL or API_TYPE == "unconfigured":
        print("ERROR: API_URL is not configured. Cannot make LLM call.")
        return {"skills": []}
        
    fallback_response = {"skills": []}

    if not text_chunk or not isinstance(text_chunk, str) or text_chunk.strip() == '':
        print("WARNING: Empty, None, or whitespace-only text chunk provided. Returning empty skills list.")
        return fallback_response

    headers = {"Content-Type": "application/json"}
    if API_TYPE == "ngrok": # Specific header for NGROK
        headers["ngrok-skip-browser-warning"] = "true"

    prompt = f"{SYSTEM_PROMPT}\n\nCV Text Segment:\n\"\"\"\n{text_chunk}\n\"\"\"\n\nBased on the instructions and the text segment above, provide the JSON output:"
    
    payload = {}
    if API_TYPE == "custom":
        payload = {
            "model": MODEL_TO_USE,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 1024} # Default options, adjust if needed
        }
    elif API_TYPE == "ollama":
        payload = {
            "model": MODEL_TO_USE,
            "prompt": prompt,
            "stream": False,
            "format": "json", 
            "options": {"temperature": 0.1, "num_predict": 1024} 
        }
    elif API_TYPE == "ngrok":
        payload = {
            "prompt": prompt, 
            "model": MODEL_TO_USE if MODEL_TO_USE else "default" # NGROK might need model in payload
        }

    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=180)
        response.raise_for_status()
        
        json_data_str = None

        if API_TYPE == "custom":
            try:
                resp_json = response.json()
                if "response" in resp_json and isinstance(resp_json["response"], str):
                    json_data_str = resp_json["response"]
                elif isinstance(resp_json, dict) and "skills" in resp_json: # Direct JSON object
                     return resp_json # Already parsed
                # If custom API might return JSON string directly in response.text and not as a JSON object field
                elif not json_data_str and response.text.strip().startswith("{"): 
                    json_data_str = response.text # Will attempt to parse later
                else:
                    print(f"ERROR: Custom LLM API response format unexpected")
                    return fallback_response
            except json.JSONDecodeError:
                 # This might happen if response.text is the JSON string itself
                if response.text.strip().startswith("{") and response.text.strip().endswith("}"):
                    json_data_str = response.text
                else:
                    print(f"ERROR: Failed to parse Custom LLM API response as JSON. Raw: {response.text[:500]}")
                    return fallback_response
        elif API_TYPE == "ollama":
            try:
                resp_json = response.json()
                if "response" in resp_json and isinstance(resp_json["response"], str):
                    json_data_str = resp_json["response"]
                elif isinstance(resp_json, dict) and "skills" in resp_json: # Direct JSON object
                     return resp_json # Already parsed
                else:
                    print(f"ERROR: Local Ollama response format unexpected. Raw: {response.text[:500]}")
                    return fallback_response
            except json.JSONDecodeError:
                print(f"ERROR: Failed to parse Ollama response as JSON. Raw: {response.text[:500]}")
                return fallback_response
        elif API_TYPE == "ngrok":
            try:
                resp_json = response.json()
                if isinstance(resp_json, dict) and 'output' in resp_json and isinstance(resp_json['output'], str):
                    json_data_str = resp_json['output']
                elif isinstance(resp_json, dict) and "skills" in resp_json: # Direct JSON object
                    return resp_json # Already parsed
                else:
                    print(f"ERROR: Hosted API (NGROK) response format unexpected")
                    return fallback_response
            except json.JSONDecodeError:
                print(f"ERROR: Failed to parse hosted API (NGROK) response as JSON. Raw: {response.text[:500]}")
                return fallback_response

        if not json_data_str:
            print("ERROR: No JSON data string found in API response.")
            return fallback_response
        
        # Clean up and parse the JSON string
        if json_data_str.strip().startswith("```json"):
            json_data_str = json_data_str.strip()[7:]
            if json_data_str.strip().endswith("```"):
                json_data_str = json_data_str.strip()[:-3]
        
        try:
            parsed_json = json.loads(json_data_str.strip())
            if isinstance(parsed_json, dict) and "skills" in parsed_json and isinstance(parsed_json["skills"], list):
                return parsed_json
            else:
                print(f"ERROR: Parsed JSON is not in the expected format {{'skills': [...]}}. Got: {parsed_json}")
                return fallback_response
        except json.JSONDecodeError:
            json_pattern = r'(\{[\s\S]*?"skills"\s*:\s*\[[\s\S]*?\][\s\S]*?\})' # More specific for {"skills": [...]}
            match = re.search(json_pattern, json_data_str)
            if match:
                try:
                    parsed_json = json.loads(match.group(1).strip())
                    if isinstance(parsed_json, dict) and "skills" in parsed_json and isinstance(parsed_json["skills"], list):
                        return parsed_json
                    else:
                        print(f"ERROR: Regex extracted JSON not in expected format. Got: {parsed_json}")
                        return fallback_response
                except json.JSONDecodeError:
                    print(f"ERROR: Failed to parse JSON extracted by regex. Content: '{match.group(1)[:500]}'")
                    return fallback_response
            else:
                print(f"ERROR: No valid JSON object {{'skills': [...]}} found in response. Raw: '{json_data_str[:500]}'")
                return fallback_response

    except requests.exceptions.HTTPError as errh:
        print(f"Http Error: {errh}")
        if errh.response is not None: print(f"Response content: {errh.response.text}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Oops: Something Else: {err}")
    except Exception as e:
        print(f"An unexpected error occurred during API call: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
    return fallback_response


def get_extracted_skills_from_file(cv_file_to_process=None):
    """
    Reads a CV file, extracts skills using LLM, and returns a list of skills.
    Args:
        cv_file_to_process (str, optional): Path to the CV file. 
                                            Defaults to CV_FILE_PATH global.
    Returns:
        list: A list of extracted skills, or None if an error occurs.
    """
    target_cv_file = cv_file_to_process if cv_file_to_process else CV_FILE_PATH
    print(f"--- Processing CV Skill Extraction from File: {sanitize_path(target_cv_file)} ---")

    if not os.path.exists(target_cv_file):
        print(f"ERROR: CV file not found at {target_cv_file}")
        return None
    
    try:
        with open(target_cv_file, 'r', encoding='utf-8') as f:
            cv_content = f.read()
        
        if not cv_content.strip():
            print("ERROR: CV file is empty.")
            return None
        
        print(f"Successfully read CV file. Content length: {len(cv_content)} characters.")
        
        cv_chunks = create_text_chunks(cv_content, MAX_CHUNK_CHAR_LENGTH, CHUNK_OVERLAP_CHAR_LENGTH)
        
        if not cv_chunks:
            print("ERROR: No text chunks could be created from the CV content.")
            return None
            
        print(f"CV content split into {len(cv_chunks)} chunk(s).")
        all_extracted_skills = set()
        
        for i, chunk in enumerate(cv_chunks):
            print(f"\nProcessing chunk {i+1}/{len(cv_chunks)} (length: {len(chunk)} chars)...")
            extracted_data_from_chunk = extract_skills_from_text(chunk)
            
            if extracted_data_from_chunk and "skills" in extracted_data_from_chunk and isinstance(extracted_data_from_chunk["skills"], list):
                if extracted_data_from_chunk["skills"]:
                    print(f"  Extracted {len(extracted_data_from_chunk['skills'])} skills from chunk {i+1}.")
                    for skill in extracted_data_from_chunk["skills"]:
                        all_extracted_skills.add(skill) 
                else:
                    print(f"  No skills found in chunk {i+1}.")
            else:
                print(f"  Failed to extract skills or unexpected format from chunk {i+1}. Received: {extracted_data_from_chunk}")

        final_skills_list = sorted(list(all_extracted_skills))
        
        if final_skills_list:
            print("\n--- Aggregated Extracted Skills (from get_extracted_skills_from_file) ---")
            # The function will return the list, printing is for when called directly
            # print(json.dumps({"skills": final_skills_list}, indent=2, ensure_ascii=False))
            return final_skills_list
        else:
            print("No skills were extracted or found in any chunk (from get_extracted_skills_from_file).")
            return [] # Return empty list if no skills found

    except Exception as e:
        print(f"An error occurred during file processing or skill extraction in get_extracted_skills_from_file: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # This part now calls the new function, preserving original script behavior
    print(f"--- Main execution of extract_skills_from_cv_file.py ---")
    extracted_skills = get_extracted_skills_from_file() # Uses default CV_FILE_PATH

    if extracted_skills is not None:
        print("\n--- Final Output from __main__ ---")
        final_output_json = {"skills": extracted_skills}
        print(json.dumps(final_output_json, indent=2, ensure_ascii=False))
    else:
        print("\n--- Main execution failed to extract skills ---")
        print(json.dumps({"skills": []}, indent=2, ensure_ascii=False))
    
    print("\n--- CV Skill Extraction Script Finished ---")