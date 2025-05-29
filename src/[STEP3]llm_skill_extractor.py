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

3.  **language_requirements**: A list of *explicitly stated* language proficiencies a candidate needs for this role.
    *   For each language mentioned, create an object: {"language": "LanguageName", "proficiency": "ProficiencyLevel"}
    *   For "ProficiencyLevel", **CHOOSE ONE from these predefined options:** ["Native/Bilingual", "Fluent", "Proficient/Business-level", "Conversational", "Basic", "Not specified"]
    *   Explicitly use only the defined ones, same capitalization, same spacing, punctuation, and wording.
    *   Example: [{"language": "English", "proficiency": "Fluent"}, {"language": "Danish", "proficiency": "Proficient/Business-level"}]
    *   If proficiency is not mentioned for an explicitly stated language required for the role, use "Not specified" for its proficiency.
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
  "language_requirements": [{"language": "Lang1", "proficiency": "ChosenProficiencyOption"}, ...],
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
CUSTOM_LLM_MODEL = os.getenv("CUSTOM_LLM_MODEL_NAME")


# Prioritize: Custom LLM > NGROK > Local Ollama
if CUSTOM_LLM_API_URL and CUSTOM_LLM_MODEL:
    API_URL = CUSTOM_LLM_API_URL

    print(f"INFO ([STEP3]llm_skill_extractor): Using Custom LLM API")
else:
    raise ValueError("CUSTOM_LLM_API_URL and CUSTOM_LLM_MODEL_NAME must be set in the environment.")

def extract_job_details_with_llm(job_text):

    fallback_response = {
        "skills": [],
        "experience_level_required": "Not specified",
        "language_requirements": [],
        "education_level_preferred": "Not specified",
        "job_type": "Not specified"
    }

    # Bail early if there's nothing to work with
    if not job_text or not isinstance(job_text, str) or job_text.strip() == '':
        print("Warning ([STEP3]llm_skill_extractor): Empty, None, or whitespace-only job text provided. Returning default structured response.")
        return fallback_response

    # Set up request headers
    headers = {"Content-Type": "application/json"}

    # Construct the full prompt for the LLM
    prompt = f"{SYSTEM_PROMPT}\n\nJob Description Text:\n\"\"\"\n{job_text}\n\"\"\"\n\nBased on the instructions and the text above, provide the JSON output:"
    payload = {
        "model": CUSTOM_LLM_MODEL,
        "prompt": prompt,
        "options": {
            "num_predict": 8192,  # Max tokens for the output
            "num_ctx": 8192,      # Context window size (input + output)
            "seed": 101,          # For reproducible outputs
            "temperature": 0.1    # Controls randomness/creativity
        },
        "stream": False
    }

    try:
        # Make the API call
        response = requests.post(API_URL, json=payload, headers=headers, timeout=180)
        response.raise_for_status()
        
        json_data = None

        try:
            resp_json = response.json()
            # The custom endpoint may return the result in a "response" field
            if "response" in resp_json and isinstance(resp_json["response"], str):
                json_data = resp_json["response"]
            # It might also return the JSON directly as the root object
            elif isinstance(resp_json, dict) and all(key in resp_json for key in fallback_response.keys()):
                 json_data = json.dumps(resp_json)
            else:
                print(f"Error ([STEP3]llm_skill_extractor): Custom API response format unexpected. Raw: {response.text[:500]}")
                return fallback_response
        except json.JSONDecodeError:
            print(f"Error ([STEP3]llm_skill_extractor): Failed to parse Custom API response as JSON. Raw: {response.text[:500]}")
            return fallback_response

        if not json_data:
            print("Error ([STEP3]llm_skill_extractor): No JSON data found in API response.")
            return fallback_response
        
        parsed_json = None
        
        # Clean up and parse the JSON string
        # Handle markdown code blocks if present
        if json_data.strip().startswith("```json"):
            json_data = json_data.strip()[7:]
            if json_data.strip().endswith("```"):
                json_data = json_data.strip()[:-3]
        
        # Try to parse the JSON directly first
        try:
            parsed_json = json.loads(json_data.strip())
        except json.JSONDecodeError:
            # If direct parsing fails, try to find a JSON object with regex
            json_pattern = r'(\{[\s\S]*?\})' 
            json_match = re.search(json_pattern, json_data)
            
            if json_match:
                json_str = json_match.group(1)
                try:
                    parsed_json = json.loads(json_str.strip())
                except json.JSONDecodeError:
                    print(f"Error ([STEP3]llm_skill_extractor): Failed to parse JSON via regex.")
                    print(f"Content tried: '{json_str[:500]}'")
                    print(f"Original JSON data: '{json_data[:500]}'")
                    return fallback_response
            else:
                print(f"Error ([STEP3]llm_skill_extractor): No JSON object found in response.")
                print(f"Raw LLM output: '{json_data[:500]}'")
                return fallback_response
        
        # Start with the fallback response and fill it in with validated data
        result = fallback_response.copy()

        if isinstance(parsed_json, dict):
            # Process skills
            skills = parsed_json.get('skills', [])
            if isinstance(skills, list):
                clean_skills = []
                seen = set()
                
                for skill in skills:
                    if isinstance(skill, str) and skill.strip():
                        clean_skill = ' '.join(skill.strip().split()).lower()
                        if clean_skill not in seen:
                            clean_skills.append(skill.strip())
                            seen.add(clean_skill)
                            
                result['skills'] = clean_skills
            
            # Process string fields
            for field in ["experience_level_required", "education_level_preferred", "job_type"]:
                value = parsed_json.get(field)
                
                if value is not None:
                    if field == "experience_level_required" and isinstance(value, list):
                        valid_values = [item for item in value if isinstance(item, str) and item.strip()]
                        if valid_values:
                            result[field] = valid_values
                        elif value:  # List exists but has only invalid items
                            result[field] = "Not specified"
                    elif value != "Not specified" and (not isinstance(value, list) or value):
                        result[field] = value

            # Process language requirements
            langs = parsed_json.get('language_requirements', [])
            if isinstance(langs, list):
                valid_langs = []
                for lang in langs:
                    if isinstance(lang, dict) and "language" in lang and isinstance(lang["language"], str) and lang["language"].strip():
                        valid_langs.append({
                            "language": lang["language"].strip(),
                            "proficiency": lang.get("proficiency", "Not specified")
                        })
                result['language_requirements'] = valid_langs
            
            return result
        else:
            print(f"Error ([STEP3]llm_skill_extractor): Parsed JSON is not a dictionary.")
            print(f"Got {type(parsed_json)}: {str(parsed_json)[:500]}")
            return fallback_response

    except requests.exceptions.HTTPError as err:
        print(f"HTTP error: {err}")
        if hasattr(err, 'response') and err.response:
            print(f"Status code: {err.response.status_code}")
            try:
                error_json = err.response.json()
                print(f"Error details: {json.dumps(error_json, indent=2)}")
            except json.JSONDecodeError:
                print(f"Response (not JSON): {err.response.text}")
        return fallback_response
    except requests.exceptions.RequestException as err:
        print(f"Request error: {err}")
        return fallback_response
    except Exception as err:
        print(f"Unexpected error: {err}")
        import traceback
        traceback.print_exc()
        return fallback_response

# Backward compatibility wrapper
def extract_skills_with_llm(job_text):
    """Old function name, now returns the full job details dict."""
    print("WARNING ([STEP3]llm_skill_extractor): Called legacy extract_skills_with_llm. Use extract_job_details_with_llm instead.")
    return extract_job_details_with_llm(job_text)
