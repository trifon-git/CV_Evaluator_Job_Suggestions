import os
import requests
import json
import re

SYSTEM_PROMPT = """
You are an expert HR Analyst and Data Extractor with 20 years of experience in meticulously analyzing job descriptions. Your primary goal is to extract structured information that defines the ideal candidate profile for a specific job opening, useful for matching.

Your TASK:
From the provided job description text, extract the following attributes that characterize the desired employee for this role. Return this information ONLY as a single, well-formed JSON object. For fields with predefined options, YOU MUST CHOOSE FROM THE PROVIDED OPTIONS or use the specified default if no information is found.

1.  **skills**: A comprehensive list of all technical skills, soft skills, tools, software, programming languages, methodologies, and other specific competencies *essential or highly desired for a candidate to succeed in this role*.
    *   Focus on abilities, knowledge, and attributes that directly contribute to performing the job effectively.
    *   Prioritize skills that distinguish a suitable candidate for *this specific position* over generic company-wide statements or values, unless those values translate directly into a demonstrable skill required for the role (e.g., 'customer-centric mindset' is a skill for a support role; 'we value teamwork' is a general statement, but 'proven ability to collaborate in a team' is a skill).
    *   Where appropriate, if a tool or technology is mentioned in the context of *required/desired experience or proficiency* (e.g., "experience with Python," "proficient in AWS," "knowledge of CI/CD tools"), capture that slightly more descriptive phrase as the skill, rather than just the bare tool name if the context enhances clarity for candidate evaluation. However, prioritize conciseness if the context doesn't add significant value.
    *   Example: Prefer "experience with Django" over just "Django" if the text says "3 years of experience with Django." Prefer "Git" if the text just says "knowledge of Git."
    *   Do NOT include full sentences, general job duties (unless they clearly imply a specific skill), job titles, or degrees here.
    *   If no specific role-defining skills are found, return an empty list: [].

2.  **experience_level_required**: The level of professional experience required or desired for a candidate to be effective in this role.
    *   **CHOOSE ONE from these predefined options:** ["Entry-level", "Junior (0-2 years)", "Mid-level (3-5 years)", "Senior (5-10 years)", "Lead (7+ years, with leadership)", "Principal/Expert (10+ years, deep expertise)", "Manager", "Director", "Executive"]
    *   Explicitly use only the defined ones, same capitalization, same spacing, punctuation, and wording.
    *   Select the option that best reflects the seniority and responsibility described. If specific years are mentioned that align with a category, prefer that category (e.g., "4 years experience" maps to "Mid-level (3-5 years)").
    *   If multiple distinct levels are clearly required for different aspects or if it's a very broad role, you may return a list of chosen options, e.g., ["Senior (5-10 years)", "Lead (7+ years, with leadership)"]. Otherwise, prefer a single best fit.
    *   If not explicitly mentioned or clearly inferable for the role, return "Not specified".

3.  **language_requirements**: A list of *explicitly stated* language names a candidate needs for this role.
    *   Return a list of strings, where each string is a language name.
    *   Example: ["English", "Danish"]
    *   If no language requirements are explicitly mentioned as necessary for the candidate, return an empty list: []. (Do not infer the primary language of the document itself here as a candidate requirement unless stated).

4.  **education_level_preferred**: The highest level of education *explicitly stated as preferred or required* for a candidate in this role.
    *   **CHOOSE ONE from these predefined options:** ["High School Diploma/GED", "Vocational Training/Apprenticeship", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "Doctorate (PhD)", "Professional Degree (e.g., MD, JD)", "Not specified"]
    *   Explicitly use only the defined ones, same capitalization, same spacing, punctuation, and wording. 
    *   If a specific field of study is mentioned (e.g., "Bachelor's in Computer Science"), you can optionally append it to the chosen option IN A COMMENT within your thought process, but the JSON value should be one of the predefined options. For the JSON output, just return the chosen education level.
    *   If not explicitly mentioned as a candidate requirement, return "Not specified".

5.  **job_type**: The type of employment *explicitly stated* for this position.
    *   **CHOOSE ONE from these predefined options:** ["Full-time", "Part-time", "Contract/Temporary", "Internship", "Freelance"]
    *   Explicitly use only the defined ones, same capitalization, same spacing, punctuation, and wording. 
    *   If not mentioned, return "Not specified". (Infer "Full-time" only if the context strongly suggests a standard permanent role *and no other job type indicators are present*).

Output format (MUST be a single JSON object with these exact top-level keys):
{
  "skills": ["Skill 1", "Skill 2", ...],
  "experience_level_required": "ChosenOption OR [\"ChosenOption1\", \"ChosenOption2\"] OR Not specified",
  "language_requirements": ["Lang1", "Lang2", ...],
  "education_level_preferred": "ChosenOption OR Not specified",
  "job_type": "ChosenOption OR Not specified"
}

IMPORTANT RULES:
1.  Your ENTIRE response must be ONLY the single JSON object described above. Do not add any conversational text before or after the JSON.
2.  Adhere strictly to the specified keys and the PREDEFINED OPTIONS for the relevant fields.
3.  If a field's information is not found in the text for a constrained field, use "Not specified" or an empty list [] as indicated for that field.
4.  Extract information *only* from the provided "Job Description Text" that pertains to the ideal candidate for the role.
5.  For "skills", focus on abilities, knowledge, and attributes the *candidate* should possess or will use to be successful in this job.
"""

# Grab API config from environment
CUSTOM_LLM_API_URL = os.getenv("CUSTOM_LLM_API_URL")
CUSTOM_LLM_MODEL = os.getenv("CUSTOM_LLM_MODEL_NAME") # Added this line, assuming it was missing based on context
NGROK_API_URL = os.getenv("NGROK_API_URL") # Assuming this is how NGROK URL is fetched
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL") # Assuming this is how Ollama URL is fetched
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME") # Assuming this is how Ollama model is fetched


API_URL = None
CURRENT_MODEL_NAME = None # To store the model name being used
API_MODE = None # To store which API mode is active: "CUSTOM", "NGROK", "OLLAMA"

if CUSTOM_LLM_API_URL and CUSTOM_LLM_MODEL:
    API_URL = CUSTOM_LLM_API_URL
    CURRENT_MODEL_NAME = CUSTOM_LLM_MODEL
    API_MODE = "CUSTOM"
    print(f"INFO ([STEP3]llm_skill_extractor): Using Custom LLM API with model: {CURRENT_MODEL_NAME}")
elif NGROK_API_URL and OLLAMA_MODEL_NAME: # Assuming NGROK might also use OLLAMA_MODEL_NAME or a specific one
    API_URL = NGROK_API_URL
    CURRENT_MODEL_NAME = OLLAMA_MODEL_NAME # Or a specific model for NGROK if different
    API_MODE = "NGROK"
    print(f"INFO ([STEP3]llm_skill_extractor): Using NGROK API with model: {CURRENT_MODEL_NAME}")
elif OLLAMA_API_URL and OLLAMA_MODEL_NAME:
    API_URL = OLLAMA_API_URL
    CURRENT_MODEL_NAME = OLLAMA_MODEL_NAME
    API_MODE = "OLLAMA"
    print(f"INFO ([STEP3]llm_skill_extractor): Using Local Ollama API with model: {CURRENT_MODEL_NAME}")
else:
    # Fallback or error if no API is configured
    print("ERROR ([STEP3]llm_skill_extractor): No LLM API configured. Please check .env variables (CUSTOM_LLM_API_URL, NGROK_API_URL, or OLLAMA_API_URL and associated model names).")
    # Depending on desired behavior, you might raise an error here or allow the script to proceed if it can handle no LLM.
    # For now, let's assume it should raise an error if no API is found, as extraction is key.
    raise ValueError("LLM API not configured in [STEP3]llm_skill_extractor.py")


def extract_job_details_with_llm(job_text):
    fallback_response = {
        "skills": [],
        "experience_level_required": "Not specified",
        "language_requirements": [],
        "education_level_preferred": "Not specified",
        "job_type": "Not specified"
    }

    if not API_URL or not CURRENT_MODEL_NAME:
        print("Error ([STEP3]llm_skill_extractor): API_URL or CURRENT_MODEL_NAME not set. Cannot make LLM call.")
        return fallback_response.copy()

    if not job_text or not isinstance(job_text, str) or not job_text.strip():
        print("Warning ([STEP3]llm_skill_extractor): Empty, None, or whitespace-only job text provided. Returning default structured response.")
        return fallback_response.copy()

    headers = {"Content-Type": "application/json"}
    prompt = f"{SYSTEM_PROMPT}\n\nJob Description Text:\n\"\"\"\n{job_text}\n\"\"\"\n\nBased on the instructions and the text above, provide the JSON output:"
    
    payload = {
        "model": CURRENT_MODEL_NAME, # Use the determined model name
        "prompt": prompt,
        "stream": False
    }
    # For Ollama, the payload structure might be slightly different if it's a direct call not through a custom/NGROK endpoint
    # If API_MODE is "OLLAMA" and it's a direct call, it might just be:
    # payload = {"model": CURRENT_MODEL_NAME, "prompt": prompt, "stream": False}
    # If your NGROK/Custom endpoint wraps Ollama, the above payload is likely fine.

    json_data_str = None # Define for broader scope in case of parsing errors
    parsed_json = None

    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=180)
        response.raise_for_status()
        
        raw_response_text = response.text # Get raw text for debugging if needed

        try:
            resp_json_outer = response.json()
            if API_MODE == "OLLAMA" and "response" in resp_json_outer and isinstance(resp_json_outer["response"], str): # Direct Ollama
                json_data_str = resp_json_outer["response"]
            elif API_MODE == "CUSTOM" and "response" in resp_json_outer and isinstance(resp_json_outer["response"], str): # Custom endpoint
                 json_data_str = resp_json_outer["response"]
            elif API_MODE == "NGROK" and "response" in resp_json_outer and isinstance(resp_json_outer["response"], str): # NGROK endpoint
                 json_data_str = resp_json_outer["response"] # Assuming NGROK also wraps it in "response"
            # Fallback if the JSON is the root object (less common for Ollama direct)
            elif isinstance(resp_json_outer, dict) and ("skills" in resp_json_outer or "language_requirements" in resp_json_outer):
                 json_data_str = json.dumps(resp_json_outer)
            else:
                print(f"Error ([STEP3]llm_skill_extractor): API response format unexpected. API Mode: {API_MODE}. Raw: {raw_response_text[:500]}")
                return fallback_response.copy()
        except json.JSONDecodeError: # This means response.json() itself failed
            print(f"Error ([STEP3]llm_skill_extractor): Failed to parse API response envelope as JSON. Raw: {raw_response_text[:500]}")
            # Attempt to extract JSON from raw_response_text if it looks like a plain JSON string
            if raw_response_text.strip().startswith("{") and raw_response_text.strip().endswith("}"):
                json_data_str = raw_response_text
            else:
                return fallback_response.copy()

        if not json_data_str:
            print("Error ([STEP3]llm_skill_extractor): No JSON data string extracted from API response.")
            return fallback_response.copy()
        
        # Clean up and parse the JSON string
        if json_data_str.strip().startswith("```json"):
            json_data_str = json_data_str.strip()[7:]
            if json_data_str.strip().endswith("```"):
                json_data_str = json_data_str.strip()[:-3]
        
        try:
            parsed_json = json.loads(json_data_str.strip())
        except json.JSONDecodeError as e_parse:
            # If direct parsing fails, try to find a JSON object with regex (more robust for messy outputs)
            json_pattern = r'(\{[\s\S]*?\})' 
            json_match = re.search(json_pattern, json_data_str)
            if json_match:
                json_str_from_regex = json_match.group(1)
                try:
                    parsed_json = json.loads(json_str_from_regex.strip())
                except json.JSONDecodeError as e_regex_parse:
                    print(f"Error ([STEP3]llm_skill_extractor): Failed to parse JSON via regex. Error: {e_regex_parse}")
                    print(f"Content tried (regex): '{json_str_from_regex[:500]}'")
                    print(f"Original JSON data string: '{json_data_str[:500]}'")
                    return fallback_response.copy()
            else: # No JSON object found even with regex
                print(f"Error ([STEP3]llm_skill_extractor): No JSON object found in response string. Error: {e_parse}")
                print(f"Raw LLM output string: '{json_data_str[:500]}'")
                return fallback_response.copy()
        
        # Initialize result HERE, after successfully obtaining parsed_json
        result = fallback_response.copy()

        if isinstance(parsed_json, dict):
            # Process skills
            skills_from_llm = parsed_json.get('skills', [])
            if isinstance(skills_from_llm, list):
                clean_skills = []
                seen_skill_names = set()
                for skill_item in skills_from_llm:
                    if isinstance(skill_item, str) and skill_item.strip():
                        processed_skill = ' '.join(skill_item.strip().split()) # Normalize whitespace
                        if processed_skill.lower() not in seen_skill_names: # Case-insensitive check for uniqueness
                            clean_skills.append(processed_skill) # Store normalized original case
                            seen_skill_names.add(processed_skill.lower())
                result['skills'] = clean_skills
            
            # Process string fields (experience, education, job_type)
            for field_key in ["experience_level_required", "education_level_preferred", "job_type"]:
                value_from_llm = parsed_json.get(field_key)
                
                if value_from_llm is not None: # If key exists in LLM output
                    if field_key == "experience_level_required" and isinstance(value_from_llm, list):
                        # Handle list for experience_level_required
                        valid_exp_values = [item for item in value_from_llm if isinstance(item, str) and item.strip()]
                        if valid_exp_values: # If list contains valid string(s)
                            result[field_key] = sorted(list(set(valid_exp_values))) # Store unique sorted list
                        # If list is empty or contains only invalid items, result[field_key] remains "Not specified" from fallback
                    elif isinstance(value_from_llm, str) and value_from_llm.strip() and value_from_llm.strip().lower() != "not specified":
                        result[field_key] = value_from_llm.strip()
                    elif isinstance(value_from_llm, str) and (not value_from_llm.strip() or value_from_llm.strip().lower() == "not specified"):
                        # If LLM returns empty string or "Not specified", keep the fallback "Not specified"
                        pass # result[field_key] already "Not specified"
                    # If LLM returns something unexpected (e.g. a list for education_level), it keeps the fallback.
            
            # Process language requirements
            langs_from_llm = parsed_json.get('language_requirements', [])
            if isinstance(langs_from_llm, list):
                valid_langs = []
                seen_lang_names = set()
                for lang_item in langs_from_llm:
                    lang_name = None
                    if isinstance(lang_item, str) and lang_item.strip():
                        lang_name = lang_item.strip()
                    elif isinstance(lang_item, dict) and "language" in lang_item and isinstance(lang_item["language"], str) and lang_item["language"].strip():
                        # Gracefully handle if LLM still sends dicts by mistake
                        lang_name = lang_item["language"].strip()
                    
                    if lang_name:
                        processed_lang_name = ' '.join(lang_name.split()) # Normalize whitespace
                        if processed_lang_name.lower() not in seen_lang_names: # Case-insensitive check
                            valid_langs.append(processed_lang_name) # Store normalized original case
                            seen_lang_names.add(processed_lang_name.lower())
                result['language_requirements'] = sorted(list(set(valid_langs))) # Store unique sorted list
            
            return result
        else:
            print(f"Error ([STEP3]llm_skill_extractor): Parsed JSON content is not a dictionary. Type: {type(parsed_json)}. Content: {str(parsed_json)[:500]}")
            return fallback_response.copy()

    except requests.exceptions.HTTPError as err_http:
        print(f"HTTP error in extract_job_details_with_llm: {err_http}")
        if hasattr(err_http, 'response') and err_http.response is not None:
            print(f"Status code: {err_http.response.status_code}")
            try:
                error_json = err_http.response.json()
                print(f"Error details: {json.dumps(error_json, indent=2)}")
            except json.JSONDecodeError:
                print(f"Response (not JSON): {err_http.response.text[:500]}")
        return fallback_response.copy()
    except requests.exceptions.RequestException as err_req:
        print(f"Request error in extract_job_details_with_llm: {err_req}")
        return fallback_response.copy()
    # Removed specific json.JSONDecodeError here as it's handled during parsing attempts above
    except Exception as err_unexpected:
        print(f"Unexpected error in extract_job_details_with_llm: {err_unexpected}")
        import traceback
        traceback.print_exc()
        return fallback_response.copy()

# Backward compatibility wrapper
def extract_skills_with_llm(job_text):
    """Old function name, now returns the full job details dict."""
    print("WARNING ([STEP3]llm_skill_extractor): Called legacy extract_skills_with_llm. Use extract_job_details_with_llm instead for clarity.")
    return extract_job_details_with_llm(job_text)
