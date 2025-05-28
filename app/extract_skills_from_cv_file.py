import os
import requests
import json
import re
from dotenv import load_dotenv
import PyPDF2
import docx
import markdown
import traceback 

# --- Configuration ---
SYSTEM_PROMPT = """
You are an expert Head HR Manager with 20 years of experience in talent acquisition and skills assessment, proficient in both English and Danish.

Your TASK:
From the provided text segment (which is a part of a CV/resume), extract ALL skills. Return ONLY a list of these skills in JSON format.

Output format (MUST be exactly this format):
{"skills": ["Skill 1", "Skill 2", "Skill 3", ...]}

IMPORTANT RULES:
1.  DO NOT return any explanatory text, metadata, or HTML.
2.  ONLY extract skills mentioned in the text segment.
3.  Focus on actual skills, abilities, technologies, methodologies, tools, programming languages, and relevant soft skills.
4.  Do NOT include job titles, company names, degrees, or general certifications (unless the certification itself implies a specific, valuable skill, e.g., "CISSP").
5.  Return an empty list if no skills are found in the segment: {"skills": []}
"""

# Chunking Configuration
# LLM_INPUT_CHAR_LIMIT is the maximum characters we want to send in one go.
# Gemma 27b typically has an 8192 token context. 1 token ~ 4 chars.
# So, 8192 tokens ~ 32768 chars. Let's be conservative.
LLM_INPUT_CHAR_LIMIT = int(os.getenv('LLM_CV_INPUT_CHAR_LIMIT', '7000')) # Max chars for a single LLM call for CV
MAX_CHUNK_CHAR_LENGTH_IF_NEEDED = 2000  # Max characters per chunk IF chunking is needed
CHUNK_OVERLAP_CHAR_LENGTH_IF_NEEDED = 300   # Characters to overlap IF chunking is needed

# --- Load Environment Variables ---
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir) 
    dotenv_path = os.path.join(project_root_dir, '.env')
    
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path, override=True)
        print(f"INFO (extract_skills_from_cv_file.py): .env file loaded from {dotenv_path}")
    else:
        alt_dotenv_path = os.path.join(current_script_dir, '.env')
        if os.path.exists(alt_dotenv_path):
            load_dotenv(dotenv_path=alt_dotenv_path, override=True)
            print(f"INFO (extract_skills_from_cv_file.py): .env file loaded from {alt_dotenv_path}")
        else:
            print(f"WARNING (extract_skills_from_cv_file.py): .env file NOT found. API calls might fail.")
except Exception as e:
    print(f"ERROR (extract_skills_from_cv_file.py) during .env loading: {e}")

CUSTOM_LLM_API_URL = os.getenv("CUSTOM_LLM_API_URL")
CUSTOM_LLM_MODEL_NAME = os.getenv("CUSTOM_LLM_MODEL_NAME")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL_NAME")
NGROK_API_URL = os.getenv("NGROK_API_URL")

API_URL = None
MODEL_TO_USE = None
API_TYPE = "unconfigured"

if CUSTOM_LLM_API_URL and CUSTOM_LLM_MODEL_NAME:
    API_URL = CUSTOM_LLM_API_URL
    MODEL_TO_USE = CUSTOM_LLM_MODEL_NAME
    API_TYPE = "custom"
    print(f"INFO (extract_skills_from_cv_file.py): Using Custom LLM API: {API_URL} with model: {MODEL_TO_USE}")
elif OLLAMA_API_URL and OLLAMA_MODEL:
    API_URL = OLLAMA_API_URL 
    MODEL_TO_USE = OLLAMA_MODEL
    API_TYPE = "ollama"
    print(f"INFO (extract_skills_from_cv_file.py): Using Local Ollama API: {API_URL} with model: {MODEL_TO_USE}")
elif NGROK_API_URL:
    API_URL = NGROK_API_URL
    MODEL_TO_USE = OLLAMA_MODEL 
    API_TYPE = "ngrok_ollama" 
    print(f"INFO (extract_skills_from_cv_file.py): Using NGROK API: {API_URL}" + (f" with model: {MODEL_TO_USE}" if MODEL_TO_USE else ""))
else:
    print("ERROR (extract_skills_from_cv_file.py): No suitable API URL is set in .env. Skill extraction will fail.")

def _read_cv_content_from_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    text = ""
    try:
        if file_extension == ".pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() or ""
        elif file_extension == ".docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file_extension == ".md":
            with open(file_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            html = markdown.markdown(md_content)
            text = re.sub(r'<[^>]+>', ' ', html) 
            text = re.sub(r'\s+', ' ', text).strip()
        elif file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            print(f"Unsupported file type for skill extraction: {file_extension}")
            return None
        return text.strip()
    except Exception as e:
        print(f"Error reading CV file {file_path}: {e}")
        traceback.print_exc()
        return None

def create_text_chunks_if_needed(text, max_len_single_call, chunk_size_if_needed, overlap_if_needed):
    if not text or not isinstance(text, str): return []
    
    if len(text) <= max_len_single_call:
        print(f"Text length ({len(text)}) is within single call limit ({max_len_single_call}). Not chunking.")
        return [text.strip()] # Return as a single chunk
    
    print(f"Text length ({len(text)}) exceeds single call limit ({max_len_single_call}). Chunking...")
    chunks = []
    start_index = 0
    text_len = len(text)
    while start_index < text_len:
        end_index = min(start_index + chunk_size_if_needed, text_len)
        chunks.append(text[start_index:end_index])
        if end_index == text_len: break
        start_index += chunk_size_if_needed - overlap_if_needed
        if start_index >= end_index : break 
        start_index = min(start_index, text_len - 1) if text_len > 0 else 0
    return [chunk for chunk in chunks if chunk.strip()]

def extract_skills_from_text_chunk(text_chunk):
    if not API_URL or API_TYPE == "unconfigured":
        print("ERROR (extract_skills_from_cv_file.py): API_URL is not configured.")
        return {"skills": []}
    
    fallback_response = {"skills": []}
    if not text_chunk or not isinstance(text_chunk, str) or not text_chunk.strip():
        return fallback_response

    headers = {"Content-Type": "application/json"}
    if API_TYPE == "ngrok_ollama":
        headers["ngrok-skip-browser-warning"] = "true"

    prompt_for_llm = f"{SYSTEM_PROMPT}\n\nCV Text Segment:\n\"\"\"\n{text_chunk}\n\"\"\"\n\nBased on the instructions and the text segment above, provide the JSON output:"
    
    payload = {}
    if API_TYPE == "custom":
        payload = {"model": MODEL_TO_USE, "prompt": prompt_for_llm, "stream": False, "options": {"temperature": 0.1, "num_predict": 2048}} # Increased num_predict
    elif API_TYPE == "ollama": 
        payload = {"model": MODEL_TO_USE, "prompt": prompt_for_llm, "stream": False, "format": "json", "options": {"num_predict": 2048}} # Increased num_predict
    elif API_TYPE == "ngrok_ollama": 
        payload = {"model": MODEL_TO_USE, "prompt": prompt_for_llm, "stream": False, "format": "json", "options": {"num_predict": 2048}} # Increased num_predict

    try:
        print(f"Calling LLM ({API_TYPE}) for skill extraction from text (length: {len(text_chunk)})...")
        response = requests.post(API_URL, json=payload, headers=headers, timeout=180) # Increased timeout slightly
        
        if response.status_code == 503:
            print(f"Http Error 503 (Service Unavailable) from LLM API: {API_URL}")
            print(f"LLM API Response content (first 500 chars): {response.text[:500]}")
            return fallback_response # Critical error, LLM service is down

        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx) other than 503
        
        json_data_str = None
        resp_data = response.json()

        if API_TYPE == "custom":
            if "response" in resp_data and isinstance(resp_data["response"], str):
                json_data_str = resp_data["response"]
            elif isinstance(resp_data, dict) and "skills" in resp_data: 
                return resp_data 
            elif response.text.strip().startswith("{"): 
                json_data_str = response.text
            else:
                print(f"ERROR (extract_skills_from_cv_file.py): Custom LLM API response format unexpected. Raw: {response.text[:500]}")
                return fallback_response
        elif API_TYPE == "ollama" or API_TYPE == "ngrok_ollama":
            if isinstance(resp_data, dict) and "skills" in resp_data: 
                return resp_data
            elif "response" in resp_data and isinstance(resp_data["response"], str): 
                json_data_str = resp_data["response"]
            else: 
                try:
                    parsed_directly = json.loads(response.text)
                    if isinstance(parsed_directly, dict) and "skills" in parsed_directly:
                        return parsed_directly
                except json.JSONDecodeError:
                    pass 
                print(f"ERROR (extract_skills_from_cv_file.py): Ollama/NGROK API response format unexpected. Raw: {response.text[:500]}")

        if not json_data_str:
            print(f"ERROR (extract_skills_from_cv_file.py): No JSON data string extracted from API response ({API_TYPE}).")
            return fallback_response
        
        if json_data_str.strip().startswith("```json"):
            json_data_str = json_data_str.strip()[7:]
            if json_data_str.strip().endswith("```"):
                json_data_str = json_data_str.strip()[:-3]
        
        try:
            parsed_json = json.loads(json_data_str.strip())
            if isinstance(parsed_json, dict) and "skills" in parsed_json and isinstance(parsed_json["skills"], list):
                return parsed_json
            else: 
                print(f"Warning (extract_skills_from_cv_file.py): Parsed JSON not in exact {{'skills': [...]}} format. Got: {parsed_json}. Trying to find 'skills' list.")
                if isinstance(parsed_json, dict) and isinstance(parsed_json.get("skills"), list): 
                    return {"skills": parsed_json["skills"]}
                elif isinstance(parsed_json, list): 
                     print(f"Warning (extract_skills_from_cv_file.py): LLM returned a list directly. Wrapping in {{'skills': [...]}}.")
                     return {"skills": parsed_json}
                print(f"ERROR (extract_skills_from_cv_file.py): Could not conform parsed JSON. Original: {json_data_str[:200]}")
                return fallback_response
        except json.JSONDecodeError:
            json_match = re.search(r'(\{[\s\S]*\})', json_data_str) 
            if json_match:
                try:
                    potential_json_str = json_match.group(1).strip()
                    parsed_json_regex = json.loads(potential_json_str)
                    if isinstance(parsed_json_regex, dict) and "skills" in parsed_json_regex and isinstance(parsed_json_regex["skills"], list):
                        return parsed_json_regex
                    elif isinstance(parsed_json_regex, dict) and isinstance(parsed_json_regex.get("skills"), list):
                         return {"skills": parsed_json_regex["skills"]}
                    elif isinstance(parsed_json_regex, list):
                         return {"skills": parsed_json_regex}
                    print(f"ERROR (extract_skills_from_cv_file.py): Regex extracted JSON not in expected format. Content: '{potential_json_str[:200]}'")
                except json.JSONDecodeError:
                    print(f"ERROR (extract_skills_from_cv_file.py): Failed to parse JSON extracted by regex. Original: '{json_data_str[:200]}'")
            else:
                print(f"ERROR (extract_skills_from_cv_file.py): No valid JSON object found in response after attempts. Raw: '{json_data_str[:200]}'")
            return fallback_response

    except requests.exceptions.HTTPError as errh: # Other HTTP errors
        print(f"Http Error from LLM API (other than 503): {errh}")
        if hasattr(errh, 'response') and errh.response is not None: print(f"LLM API Response content: {errh.response.text[:500]}")
    except requests.exceptions.RequestException as err:
        print(f"Request Exception with LLM API: {err}")
    except Exception as e:
        print(f"An unexpected error occurred during LLM call: {e}")
        traceback.print_exc()
    return fallback_response

def get_extracted_skills_from_file(cv_file_path):
    print(f"--- Starting Skill Extraction from File: {cv_file_path} ---")

    cv_content = _read_cv_content_from_file(cv_file_path)
    if not cv_content:
        print(f"ERROR (extract_skills_from_cv_file.py): Could not read content from {cv_file_path}.")
        return None 
    
    print(f"Successfully read CV file. Content length: {len(cv_content)} characters.")
    
    # Use the new conditional chunking function
    cv_chunks = create_text_chunks_if_needed(
        cv_content, 
        LLM_INPUT_CHAR_LIMIT, 
        MAX_CHUNK_CHAR_LENGTH_IF_NEEDED, 
        CHUNK_OVERLAP_CHAR_LENGTH_IF_NEEDED
    )

    if not cv_chunks:
        print("ERROR (extract_skills_from_cv_file.py): No text chunks created from CV content.")
        return [] 
        
    print(f"CV content prepared into {len(cv_chunks)} chunk(s) for LLM processing.")
    all_extracted_skills_set = set()
    
    for i, chunk in enumerate(cv_chunks):
        # The chunk here is now either the full CV (if short enough) or one of the smaller chunks
        print(f"\nProcessing text segment {i+1}/{len(cv_chunks)} (length: {len(chunk)} chars) with LLM...")
        extracted_data = extract_skills_from_text_chunk(chunk) # This function now handles a single text segment
        
        chunk_skills = extracted_data.get("skills", [])
        if isinstance(chunk_skills, list) and chunk_skills:
            print(f"  Extracted {len(chunk_skills)} skills from segment {i+1}: {chunk_skills[:5]}...") 
            for skill in chunk_skills:
                if isinstance(skill, str) and skill.strip():
                    all_extracted_skills_set.add(skill.strip()) 
        elif not chunk_skills:
            print(f"  No skills found by LLM in segment {i+1}.")
        else:
            print(f"  Warning: LLM response for segment {i+1} had 'skills' but not a list, or was malformed. Data: {extracted_data}")

    final_skills_list = sorted(list(all_extracted_skills_set))
    
    if final_skills_list:
        print(f"\n--- Aggregated {len(final_skills_list)} Unique Skills Extracted ---")
    else:
        print("\n--- No skills were extracted or found in any text segment. ---")
    return final_skills_list

if __name__ == "__main__":
    print(f"--- Main execution of extract_skills_from_cv_file.py for testing ---")
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    test_cv_dir = os.path.join(os.path.dirname(current_script_dir), "data") # Assumes 'data' is sibling to 'app'
    default_test_cv_path = os.path.join(test_cv_dir, "cv_text_example.md") 

    if not os.path.exists(default_test_cv_path):
        print(f"Test CV file not found at: {default_test_cv_path}")
        if not os.path.exists(test_cv_dir): os.makedirs(test_cv_dir, exist_ok=True)
        with open(default_test_cv_path, "w", encoding="utf-8") as f:
            f.write("Test CV with skills in Python, Java, and communication.\nAlso experienced with project management.")
        print(f"Created a dummy test file at: {default_test_cv_path}")

    extracted_skills = get_extracted_skills_from_file(default_test_cv_path)

    if extracted_skills is not None: 
        print("\n--- Final Output from __main__ Test ---")
        final_output_json = {"skills": extracted_skills}
        print(json.dumps(final_output_json, indent=2, ensure_ascii=False))
        print(f"Total unique skills extracted: {len(extracted_skills)}")
    else:
        print("\n--- Main execution test failed (file read or processing error). ---")
    
    print("\n--- CV Skill Extraction Script Finished ---")
