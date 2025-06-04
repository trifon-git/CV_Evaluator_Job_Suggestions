import os
import logging
from typing import Optional
import openai
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  # This will look for .env in the current directory or parent directories

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoverLetterGenerator:
    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize the CoverLetterGenerator with OpenAI API.
        """
        try:
            openai.api_key = os.getenv("OPENAI_API")
            self.model_name = model_name
            logger.info(f"OpenAI API key loaded and model {model_name} selected.")
        except Exception as e:
            logger.error(f"Error setting up OpenAI client: {type(e).__name__}")
            raise

    def generate_cover_letter(
            self,
            job_description: str,
            cv_text: str,
            temperature: float = 0.7,
            max_tokens: int = 800
        ) -> Optional[str]:
        try:
            if not job_description or not cv_text:
                raise ValueError("Job description and CV text must not be empty.")

            prompt = f"""
You are an expert job application assistant.

Below is a candidate's CV and a job description.

Generate a professional, 3-paragraph cover letter tailored to the job. Start the letter with "Dear Hiring Manager," and end it with "Sincerely, [Your Name]". Use formal tone and ensure the letter is directly relevant to the job.

---

CV:
{cv_text}

---

Job Description:
{job_description}

---

Cover Letter:
"""

            logger.info("Sending prompt to OpenAI...")
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95,
                n=1
            )

            generated_text = response["choices"][0]["message"]["content"].strip()

            # Sanitize
            sanitized = re.sub(r'[^\x00-\x7F]+', '', generated_text)
            sanitized = re.sub(r'(\r\n|\r|\n){3,}', '\n\n', sanitized)
            sanitized = re.sub(r'[ \t]{2,}', ' ', sanitized)

            return sanitized

        except Exception as e:
            logger.error(f"Error generating cover letter: {type(e).__name__}")
            return None


def main():
    try:
        generator = CoverLetterGenerator()

        job_desc = """
        We're hiring a Data Scientist with strong Python skills, proficiency in machine learning and NLP techniques.
        Experience with model deployment and visualization tools is a plus.
        """

        cv_text = """
        Experienced ML engineer with 3+ years working in Python, PyTorch, and NLP. 
        Developed production-ready ML pipelines, deployed models on cloud infrastructure.
        Master's degree in Data Science.
        """

        letter = generator.generate_cover_letter(job_desc, cv_text)
        if letter:
            print("\nGenerated Cover Letter:\n" + "-"*50)
            print(letter)
            print("-"*50)
        else:
            print("❌ Cover letter generation failed.")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
