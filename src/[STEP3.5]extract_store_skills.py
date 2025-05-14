import os
import sys
import importlib.util
from datetime import datetime
from pymongo import MongoClient
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection details
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('MONGO_DB_NAME')
COLLECTION_NAME = os.getenv('MONGO_COLLECTION')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10'))
HTML_CHUNK_SIZE = int(os.getenv('HTML_CHUNK_SIZE', 500)) # Max characters per chunk for LLM

def log_message(message):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {message}")

def import_module_with_brackets(file_path, module_name):
    """Import a module with brackets in the filename."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

skill_extractor_path = os.path.join(os.path.dirname(__file__), '[STEP3]llm_skill_extractor.py')
skill_extractor = import_module_with_brackets(skill_extractor_path, 'skill_extractor')
extract_skills_with_llm = skill_extractor.extract_skills_with_llm

def chunk_html_content(html_content, chunk_size):
    """Splits HTML content into chunks of a specified size."""
    chunks = []
    for i in range(0, len(html_content), chunk_size):
        chunks.append(html_content[i:i + chunk_size])
    return chunks

def init_mongodb():
    """Initialize MongoDB connection."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    return client, collection

def extract_and_store_skills(job_id, html_content, collection):
    """Extract skills from HTML content (in chunks) and store in MongoDB."""
    try:
        log_message(f"[Job ID: {job_id}] Extracting skills from HTML content (chunking enabled)")
        
        if not html_content:
            log_message(f"[Job ID: {job_id}] HTML content is empty. Skipping skill extraction.")
            return False

        log_message(f"[Job ID: {job_id}] HTML content length: {len(html_content)} characters")

        html_chunks = chunk_html_content(html_content, HTML_CHUNK_SIZE)
        

        all_extracted_skills = set() # Use a set to store unique skills

        log_message(f"[Job ID: {job_id}] Processing {len(html_chunks)} HTML chunks.")

        for i, chunk in enumerate(html_chunks):
            log_message(f"[Job ID: {job_id}] Processing chunk {i+1}/{len(html_chunks)}")
            skills_data = extract_skills_with_llm(chunk)
            
            if skills_data and isinstance(skills_data, dict) and 'skills' in skills_data:
                skills_list_from_chunk = skills_data['skills']
                if isinstance(skills_list_from_chunk, list):
                    for skill in skills_list_from_chunk:
                        if isinstance(skill, str): # Ensure skill is a string
                             all_extracted_skills.add(skill.strip())
                    log_message(f"[Job ID: {job_id}] Extracted {len(skills_list_from_chunk)} skills from chunk {i+1}")
                else:
                    log_message(f"[Job ID: {job_id}] 'skills' in skills_data is not a list for chunk {i+1}. Data: {skills_list_from_chunk}")
            else:
                log_message(f"[Job ID: {job_id}] Invalid or no skills data from LLM for chunk {i+1}. Data: {skills_data}")

        if all_extracted_skills:
            final_skills_list = sorted(list(all_extracted_skills)) 
            collection.update_one(
                {'_id': job_id},
                {'$set': {
                    'Skills': final_skills_list
                }}
            )
            log_message(f"[Job ID: {job_id}] Successfully stored {len(final_skills_list)} unique skills from {len(html_chunks)} chunks.")
            return True
        else:
            log_message(f"[Job ID: {job_id}] No skills extracted after processing all chunks.")

            collection.update_one(
                {'_id': job_id},
                {'$set': {'Skills': []}} 
            )
            return False
            
    except Exception as e:
        log_message(f"[Job ID: {job_id}] Error extracting skills with chunking: {str(e)}")
        return False

def process_jobs_with_html_content(batch_size=None, process_all=False):
    """Process jobs that have HTML content but no skills extracted yet."""
    if batch_size is None:
        batch_size = BATCH_SIZE
        
    try:
        client, collection = init_mongodb()
        
        # Modified query to only check for HTML content existence
        query = {
            'html_content': {'$exists': True, '$ne': None},
            'Skills': {'$exists': False},  # Only process jobs without skills
            'Status': 'active'  # Only process active jobs
        }
        log_message("Processing active jobs with HTML content but no skills.")
        
        total_jobs = collection.count_documents(query)
        log_message(f"Found {total_jobs} active jobs to process")
        
        if total_jobs == 0:
            log_message("No active jobs to process. Exiting.")
            return
        
        # Process jobs in batches
        processed_count = 0
        successful_count = 0
        
        # Create progress bar
        progress_bar = tqdm(total=total_jobs, desc="Extracting skills", unit="job")
        
        while processed_count < total_jobs:
            # Get a batch of jobs
            jobs = list(collection.find(query).limit(batch_size))
            
            if not jobs:
                break
                
            for job in jobs:
                job_id = job['_id']
                html_content = job['html_content']
                
                # Extract and store skills
                if extract_and_store_skills(job_id, html_content, collection):
                    successful_count += 1
                
                processed_count += 1
                progress_bar.update(1)
            
            log_message(f"Processed {processed_count}/{total_jobs} jobs, {successful_count} successful")
        
        progress_bar.close()
        log_message(f"Completed processing {processed_count} active jobs. Successfully extracted skills for {successful_count} jobs.")
        
    except Exception as e:
        log_message(f"Error processing jobs: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    log_message("Starting skill extraction for active jobs with HTML content...")
    process_jobs_with_html_content(process_all=False)