import os
import ollama
import json
import re

# Simplified and more compact system prompt
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


def extract_skills_with_llm(cv_text):
    """Extract all skills from CV text using LLM into a single list."""
    default_response = {"skills": []}

    if not cv_text.strip():
        print("Warning: Empty CV text provided. Returning default empty skills.")
        return default_response

    try:
        response = ollama.generate(
            model='llama3.2:latest',  # Changed from llama3.2:latest to qwen3:8b
            prompt=cv_text,
            system=SYSTEM_PROMPT,
            format='json',
            options={"temperature": 0.1}
        )
        
        extracted_data_str = response.get('response')
        
        if not extracted_data_str:
            print("Error: LLM returned an empty response string.")
            return default_response

        try:
            # Clean response if needed
            if extracted_data_str.strip().startswith("```json"):
                extracted_data_str = extracted_data_str.strip()[7:]
                if extracted_data_str.strip().endswith("```"):
                    extracted_data_str = extracted_data_str.strip()[:-3]
            
            # Remove any non-JSON text before or after the JSON object
            json_pattern = r'(\{.*\})'
            json_match = re.search(json_pattern, extracted_data_str, re.DOTALL)
            if json_match:
                extracted_data_str = json_match.group(1)
            
            extracted_skills_data = json.loads(extracted_data_str.strip())
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response from LLM: {e}")
            print(f"LLM raw response string (attempted clean): '{extracted_data_str}'")
            return default_response

        # Validate the structure of the parsed JSON
        if isinstance(extracted_skills_data, dict) and \
           'skills' in extracted_skills_data and \
           isinstance(extracted_skills_data['skills'], list):
            
            # Ensure all skills are strings and remove duplicates
            skills_list = extracted_skills_data['skills']
            cleaned_skills = []
            
            for skill in skills_list:
                if isinstance(skill, str) and skill.strip():
                    # Normalize skill (lowercase, remove extra spaces)
                    normalized_skill = ' '.join(skill.strip().split())
                    if normalized_skill not in [s.lower() for s in cleaned_skills]:
                        cleaned_skills.append(skill.strip())
            
            print(f"Successfully extracted {len(cleaned_skills)} skills.")
            return {"skills": cleaned_skills}
        else:
            print(f"Error: Parsed JSON does not match expected structure.")
            print(f"Received structure: {json.dumps(extracted_skills_data, indent=2)}")
            return default_response

    except Exception as e:
        print(f"An error occurred during LLM skill extraction: {e}")
        return default_response

# For backward compatibility
def extract_all_skills_combined(cv_text):
    """Legacy function that returns the same result as extract_skills_with_llm now."""
    result = extract_skills_with_llm(cv_text)
    return result.get('skills', [])