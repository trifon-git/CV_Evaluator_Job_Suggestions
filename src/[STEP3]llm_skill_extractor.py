import os
import requests
import json
import re

SYSTEM_PROMPT = """
You are a Head HR Manager with 20 years of experience in talent acquisition and skills assessment.

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

NGROK_API_URL = os.getenv("NGROK_API_URL")

if NGROK_API_URL is None:
    raise ValueError("NGROK_API_URL environment variable not set. Please set it in your .env file or environment.")


def extract_skills_with_llm(cv_text):
    """Extract all skills from CV text using the hosted LLM via ngrok."""
    default_response = {"skills": []}

    if not cv_text.strip():
        print("Warning: Empty CV text provided. Returning default empty skills.")
        return default_response

    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
    }

    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Job Description Text:\n\"\"\"\n{cv_text}\n\"\"\"\n\n"
        f"Based on the instructions and the text above, provide the JSON output:"
    )
    
    payload = {
        "prompt": full_prompt
    }
    
    print(f"Sending payload to LLM: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(NGROK_API_URL, json=payload, headers=headers, timeout=120)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        
        # First, parse the outer JSON response from the API
        try:
            outer_response_data = response.json()
        except json.JSONDecodeError as e:
            print(f"Error parsing the initial JSON response from LLM: {e}")
            print(f"Hosted LLM raw response string: '{response.text}'")
            return default_response

        # Check if 'output' key exists and its value is a string
        if isinstance(outer_response_data, dict) and 'output' in outer_response_data and isinstance(outer_response_data['output'], str):
            extracted_data_str = outer_response_data['output']
            print(f"Extracted 'output' field content: '{extracted_data_str}'")
        else:
            print("Error: LLM response does not contain an 'output' field with a string value.")
            print(f"Received structure: {json.dumps(outer_response_data, indent=2)}")
            return default_response

        if not extracted_data_str:
            print("Error: Hosted LLM 'output' field is an empty string.")
            return default_response

        # Now, process the extracted_data_str to find the skills JSON
        try:
            # Clean response if needed (applies to the content of 'output')
            if extracted_data_str.strip().startswith("```json"):
                extracted_data_str = extracted_data_str.strip()[7:]
                if extracted_data_str.strip().endswith("```"):
                    extracted_data_str = extracted_data_str.strip()[:-3]
            
            # Use regex to find the JSON object within the 'output' string
            json_pattern = r'(\{[\s\S]*?\})' # More robust regex to capture nested structures
            json_match = re.search(json_pattern, extracted_data_str)

            if json_match:
                json_str_from_output = json_match.group(1)
                print(f"Found JSON string in 'output': '{json_str_from_output}'")
                extracted_skills_data = json.loads(json_str_from_output.strip())
            else:
                print(f"Error: No JSON object found within the LLM 'output' field.")
                print(f"Content of 'output' field after cleaning: '{extracted_data_str}'")
                return default_response
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from the 'output' field of LLM response: {e}")
            print(f"Content of 'output' field (attempted clean): '{extracted_data_str}'")
            return default_response

        if isinstance(extracted_skills_data, dict) and \
           'skills' in extracted_skills_data and \
           isinstance(extracted_skills_data['skills'], list):
            
            skills_list = extracted_skills_data['skills']
            cleaned_skills = []
            
            for skill in skills_list:
                if isinstance(skill, str) and skill.strip():
                    normalized_skill = ' '.join(skill.strip().split())
                    if not any(s.lower() == normalized_skill.lower() for s in cleaned_skills):
                        cleaned_skills.append(skill.strip())
            
            print(f"Successfully extracted {len(cleaned_skills)} unique skills using hosted LLM.")
            return {"skills": cleaned_skills}
        else:
            print(f"Error: Parsed JSON from 'output' field does not match expected skills structure.")
            print(f"Received skills structure: {json.dumps(extracted_skills_data, indent=2)}")
            return default_response

    except requests.exceptions.HTTPError as http_err: # More specific error handling for HTTP errors
        print(f"HTTP error occurred: {http_err}")
        if http_err.response is not None:
            print(f"Response status code: {http_err.response.status_code}")
            try:
                # FastAPI often provides detailed validation errors in the JSON response body for 422
                error_details = http_err.response.json()
                print(f"Response content (JSON): {json.dumps(error_details, indent=2)}")
            except json.JSONDecodeError:
                print(f"Response content (not JSON): {http_err.response.text}")
        return default_response
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred during the request to the hosted LLM: {req_err}")
        return default_response
    except Exception as e:
        print(f"An unexpected error occurred during hosted LLM skill extraction: {e}")
        return default_response

# For backward compatibility
def extract_all_skills_combined(cv_text):
    """Legacy function that returns the same result as extract_skills_with_llm now."""
    result = extract_skills_with_llm(cv_text)
    return result.get('skills', [])