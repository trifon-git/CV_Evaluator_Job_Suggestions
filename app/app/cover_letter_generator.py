import os
import logging
from typing import Optional
import openai 
import re
from dotenv import load_dotenv
import traceback
# Load environment variables (ensure .env is in the project root or adjust path)
# For Streamlit deployment, secrets are usually handled by the platform
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(project_root, '.env')
if not load_dotenv(dotenv_path=dotenv_path) and not os.getenv("OPENAI_API"):
    print("Warning: .env file not found or OPENAI_API key not set directly as environment variable.")
    print(f"Attempted .env path: {dotenv_path}")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoverLetterGenerator:
    def __init__(self, model_name: str = "gpt-3.5-turbo"): # Using a cheaper model by default
        """
        Initialize the CoverLetterGenerator.
        Ensure OPENAI_API environment variable is set.
        """
        self.api_key = os.getenv("OPENAI_API")
        if not self.api_key:
            logger.error("OPENAI_API environment variable not found.")
            raise ValueError("OPENAI_API key is not set.")
        
        # For openai < 1.0
        openai.api_key = self.api_key
        # For openai >= 1.0
        # self.client = openai.OpenAI(api_key=self.api_key)

        self.model_name = model_name
        logger.info(f"CoverLetterGenerator initialized with model {model_name}.")

    def generate_cover_letter(
            self,
            job_description: str,
            cv_text: str,
            temperature: float = 0.7,
            max_tokens: int = 800 # Adjusted for typical cover letter length
        ) -> Optional[str]:
        try:
            if not job_description or not cv_text:
                logger.error("Job description and CV text must not be empty.")
                return None # Or raise ValueError

            prompt = f"""
You are an expert job application assistant. Your task is to draft a concise and professional 3-paragraph cover letter.
The letter should be tailored specifically to the job description provided, using relevant information from the candidate's CV.
Address the letter "Dear Hiring Manager," and conclude with "Sincerely, [Your Name]".
Maintain a formal tone throughout. Focus on highlighting how the candidate's skills and experience directly align with the job requirements.

--- START OF CANDIDATE CV ---
{cv_text}
--- END OF CANDIDATE CV ---

--- START OF JOB DESCRIPTION ---
{job_description}
--- END OF JOB DESCRIPTION ---

Based on the CV and Job Description, please generate the cover letter now:
Cover Letter:
"""

            logger.info(f"Sending prompt to OpenAI model {self.model_name} for cover letter generation...")
            
            # For openai < 1.0
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95, # Consider adjusting or removing top_p if using temperature
                n=1
            )
            generated_text = response["choices"][0]["message"]["content"].strip()

            # For openai >= 1.0
            # response = self.client.chat.completions.create(
            #     model=self.model_name,
            #     messages=[{"role": "user", "content": prompt}],
            #     temperature=temperature,
            #     max_tokens=max_tokens,
            #     top_p=0.95,
            #     n=1
            # )
            # generated_text = response.choices[0].message.content.strip()


            # Basic sanitation
            sanitized_text = re.sub(r'[^\x00-\x7F]+', '', generated_text) # Remove non-ASCII characters
            sanitized_text = re.sub(r'(\r\n|\r|\n){3,}', '\n\n', sanitized_text) # Normalize multiple newlines
            sanitized_text = re.sub(r'[ \t]{2,}', ' ', sanitized_text) # Normalize multiple spaces/tabs

            logger.info("Cover letter generated successfully.")
            return sanitized_text

        except openai.APIError as e: # Catch OpenAI specific errors
            logger.error(f"OpenAI API Error generating cover letter: {e}")
            # You might want to check e.status_code or e.message for more details
            return f"Error from OpenAI: {e}"
        except Exception as e:
            logger.error(f"Unexpected error generating cover letter: {str(e)}")
            traceback.print_exc()
            return None


def main_test(): # Renamed to avoid conflict if app.py also has main()
    logger.info("Starting CoverLetterGenerator test...")
    try:
        # Ensure API key is loaded for testing
        if not os.getenv("OPENAI_API"):
            print("Skipping test: OPENAI_API key not set. Please set it in your .env file.")
            return

        generator = CoverLetterGenerator()

        sample_job_desc = """
        We are seeking a motivated Software Engineer to join our dynamic team.
        Responsibilities include developing and maintaining web applications, collaborating with cross-functional teams,
        and contributing to all phases of the development lifecycle.
        Required skills: Proficiency in Python, experience with Django or Flask, familiarity with RESTful APIs,
        and strong problem-solving abilities. BSc in Computer Science or related field.
        """

        sample_cv_text = """
        Lauris Piziks
        Software Developer
        Highly skilled and results-oriented Software Developer with 3 years of experience in Python development,
        specializing in building scalable web applications using Flask and Django.
        Proven ability to design and implement RESTful APIs and integrate third-party services.
        Adept at problem-solving and working in agile environments.
        Education: BSc in Software Engineering.
        Technical Skills: Python, Flask, Django, PostgreSQL, Docker, Git, JavaScript, HTML, CSS.
        """

        print("\nGenerating cover letter for sample job and CV...")
        letter = generator.generate_cover_letter(sample_job_desc, sample_cv_text)
        
        if letter:
            print("\n--- Generated Cover Letter ---")
            print(letter)
            print("----------------------------")
        else:
            print("\n❌ Cover letter generation failed during test.")

    except ValueError as ve: # Catch init error if API key is missing
        print(f"Test Initialization Error: {ve}")
    except Exception as e:
        logger.error(f"Error in CoverLetterGenerator main_test: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main_test()
