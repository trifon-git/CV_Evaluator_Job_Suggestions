import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pymongo import MongoClient
from chromadb import HttpClient
from chromadb.config import Settings
import time
import requests
import urllib3

load_dotenv()

# Configuration
MODEL_NAME_FOR_EMBEDDING = os.getenv('MODEL_NAME', 'paraphrase-multilingual-mpnet-base-v2')
MONGO_URI = os.getenv('MONGO_URI')
CHROMA_HOST = os.getenv('CHROMA_HOST')
CHROMA_PORT = int(os.getenv('CHROMA_PORT', '8000'))
COLLECTION_NAME = os.getenv('CHROMA_COLLECTION')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10'))
VERIFY_SSL = os.getenv('VERIFY_SSL', 'true').lower() == 'true'

EMBEDDING_API_URL = os.getenv('EMBEDDING_API_URL')
USE_REMOTE_EMBEDDING = os.getenv('USE_REMOTE_EMBEDDING', 'false').lower() == 'true'
OUTPUT_EMBEDDINGS_FILE = None  

# Disable SSL warnings if VERIFY_SSL is set to false
if os.getenv('VERIFY_SSL', 'true').lower() != 'true':
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Helper Functions ---
def log_message(message):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {message}")

def chunk_list(lst, chunk_size):
    """Split a list into chunks of specified size"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def get_remote_embeddings(texts):
    """Call remote API to get embeddings for texts."""
    if not EMBEDDING_API_URL:
        log_message("Error: EMBEDDING_API_URL not configured for remote embeddings.")
        return []
    try:
        log_message(f"Calling remote embedding API at {EMBEDDING_API_URL}")
        response = requests.post(EMBEDDING_API_URL, json={"texts": texts}, verify=VERIFY_SSL)
        response.raise_for_status()
        embeddings = response.json().get("embeddings", [])
        if not embeddings:
            log_message("Warning: Empty embedding response from API")
            return []
        return embeddings
    except Exception as e:
        log_message(f"Error calling embedding API: {e}")
        return []

def generate_average_skill_embedding(model, skills_list):
    """
    Generates an average embedding for a list of skill strings.
    """
    if not skills_list or not isinstance(skills_list, list):
        return None
    
    # Filter out any empty or whitespace-only skills
    valid_skills = [skill for skill in skills_list if isinstance(skill, str) and skill.strip()]
    if not valid_skills:
        return None

    try:
        skill_embeddings = model.encode(valid_skills, show_progress_bar=False)
        if skill_embeddings is None or len(skill_embeddings) == 0:
            return None
        
        average_embedding = np.mean(skill_embeddings, axis=0)
        
        # Normalize the average embedding (optional but often good practice)
        norm = np.linalg.norm(average_embedding)
        if norm > 0:
            average_embedding = average_embedding / norm
            
        return average_embedding.tolist()  # Convert to list for JSON serialization
    except Exception as e:
        log_message(f"ERROR generating embeddings for skills: {valid_skills} - {e}")
        return None

def save_to_verification_file(job_id, job_data, embedding_info):
    """Save job details to verification file"""
    record = {
        "job_id": job_id,
        "title": job_data.get("Title", ""),
        "company": job_data.get("Company", ""),
        "skills": job_data.get("skills", []),
        "embedding_info": {
            "embedding_source": embedding_info.get("source", "local"),
            "embedding_dimensions": len(embedding_info.get("embedding", [])),
            "timestamp": datetime.now().isoformat()
        }
    }
    
    try:
        with open(OUTPUT_EMBEDDINGS_FILE, 'a', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')
    except Exception as e:
        log_message(f"Error writing to verification file: {e}")

def process_mongodb_skills():
    global OUTPUT_EMBEDDINGS_FILE
    
    # Set up output file
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    OUTPUT_EMBEDDINGS_FILE = os.path.join(data_dir, f"mongodb_skills_embeddings_{timestamp_str}.jsonl")
    
    log_message(f"Output will be saved to: {OUTPUT_EMBEDDINGS_FILE}")

    # Connect to MongoDB
    try:
        log_message(f"Connecting to MongoDB")
        mongo_client = MongoClient(MONGO_URI)
        mongo_db = mongo_client["job_scraper"]
        jobs_collection = mongo_db["jobs"]
        log_message("Successfully connected to MongoDB")
    except Exception as e:
        log_message(f"ERROR connecting to MongoDB: {e}")
        return

    # Connect to ChromaDB
    try:
        log_message(f"Connecting to ChromaDB")
        chroma_client = HttpClient(
            host=CHROMA_HOST, 
            port=CHROMA_PORT,
            ssl=False,
            headers={"accept": "application/json", "Content-Type": "application/json"}
        )
        chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
        log_message(f"Successfully connected to ChromaDB collection: {COLLECTION_NAME}")
        
        # Get existing IDs from ChromaDB
        existing_ids = set(chroma_collection.get()['ids'])
        log_message(f"Found {len(existing_ids)} existing embeddings in ChromaDB collection '{COLLECTION_NAME}'")
    except Exception as e:
        log_message(f"ERROR connecting to ChromaDB: {e}")
        return

    # Query jobs with skills
    query = {
        "skills": {"$exists": True, "$ne": []},
        "Status": "active"
    }
    
    try:
        # Count total jobs for progress bar
        total_jobs = jobs_collection.count_documents(query)
        log_message(f"Found {total_jobs} jobs with skills in MongoDB")
        
        if total_jobs == 0:
            log_message("No jobs with skills found. Exiting.")
            return
        
        # Initialize local embedding model (only if needed)
        local_embedding_model = None
        
        # Process jobs in batches
        jobs = [job for job in jobs_collection.find(query) if str(job.get("_id")) not in existing_ids]
        log_message(f"Found {len(jobs)} new active jobs with skills to process")
        
        # Process jobs in batches
        for batch in tqdm(chunk_list(jobs, BATCH_SIZE), total=len(jobs) // BATCH_SIZE + 1, desc="Embedding new jobs"):
            ids = []
            embeddings = []
            metadatas = []
            
            # Prepare batch data
            batch_skills = []
            batch_jobs = []
            
            for job in batch:
                try:
                    job_id = str(job.get("_id"))
                    skills = job.get("skills", [])
                    
                    # Skip jobs with no skills
                    if not skills:
                        continue
                    
                    # Add to batch for processing
                    batch_jobs.append(job)
                    batch_skills.append(skills)
                    
                except Exception as e:
                    log_message(f"Error processing job {job.get('_id')}: {e}")
            
            if not batch_jobs:
                continue
                
            # Try remote embedding first if configured
            embedding_source = "local"  # Default source
            batch_embeddings = None
            
            if USE_REMOTE_EMBEDDING and EMBEDDING_API_URL:
                log_message(f"Attempting remote embedding for batch...")
                # Convert skills lists to strings for API
                skill_texts = [", ".join(skills) for skills in batch_skills]
                remote_embeddings = get_remote_embeddings(skill_texts)
                
                if remote_embeddings and len(remote_embeddings) == len(batch_jobs):
                    batch_embeddings = remote_embeddings
                    embedding_source = "remote"
                    log_message(f"✓ Successfully fetched {len(batch_embeddings)} embeddings remotely.")
                else:
                    log_message("⚠️ Remote embedding failed or returned unexpected result. Will use local embedding.")
                    if remote_embeddings is None or not remote_embeddings:
                        log_message("   Reason: No embeddings returned from remote API.")
                    elif len(remote_embeddings) != len(batch_jobs):
                        log_message(f"   Reason: Mismatch in count. Expected {len(batch_jobs)}, got {len(remote_embeddings)}.")
            
            # Use local embedding if remote failed or wasn't used
            if batch_embeddings is None:
                # Load the embedding model if not already loaded
                if local_embedding_model is None:
                    log_message(f"Loading sentence transformer model: {MODEL_NAME_FOR_EMBEDDING}")
                    try:
                        local_embedding_model = SentenceTransformer(MODEL_NAME_FOR_EMBEDDING)
                        log_message("Embedding Model loaded successfully.")
                    except Exception as e:
                        log_message(f"ERROR loading sentence transformer model: {e}")
                        return
                
                # Generate embeddings locally
                log_message("Generating embeddings locally...")
                batch_embeddings = []
                for skills in batch_skills:
                    skill_embedding = generate_average_skill_embedding(local_embedding_model, skills)
                    if skill_embedding:
                        batch_embeddings.append(skill_embedding)
                    else:
                        batch_embeddings.append(None)  # Placeholder for failed embedding
            
            # Process results and prepare for ChromaDB
            for i, (job, embedding) in enumerate(zip(batch_jobs, batch_embeddings)):
                if embedding is None:
                    continue
                    
                job_id = str(job.get("_id"))
                skills = job.get("skills", [])
                
                ids.append(job_id)
                embeddings.append(embedding)
                
                # Prepare metadata
                metadata = {
                    "Title": job.get("Title", ""),
                    "URL": job.get("URL", ""),
                    "Application_URL": job.get("Application_URL", ""),
                    "Created_At": str(job.get("Created_At", "")),
                    "Published_Date": str(job.get("Published_Date", "")) if job.get("Published_Date") else None,
                    "Company": job.get("Company", ""),
                    "Area": job.get("Area", ""),
                    "Category": job.get("Category", ""),
                    "Status": job.get("Status", "active"),
                    "skills": ", ".join(skills)  # Convert skills list to comma-separated string
                }
                metadatas.append(metadata)
                
                # Save to verification file
                save_to_verification_file(
                    job_id, 
                    {
                        "Title": job.get("Title", ""),
                        "Company": job.get("Company", ""),
                        "skills": skills
                    }, 
                    {"source": embedding_source, "embedding": embedding}
                )
            
            # Add to ChromaDB if we have embeddings
            if ids and embeddings and len(ids) == len(embeddings) == len(metadatas):
                try:
                    chroma_collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
                    log_message(f"✓ Added {len(ids)} jobs to ChromaDB for this batch.")
                except Exception as e:
                    log_message(f"❌ Failed to add batch to ChromaDB: {e}")
            
            # Add a small delay to avoid overwhelming ChromaDB
            time.sleep(0.5)
        
        log_message(f"Finished processing. Successfully saved embeddings to ChromaDB.")
        log_message(f"Verification file saved to: {OUTPUT_EMBEDDINGS_FILE}")
        
    except Exception as e:
        log_message(f"ERROR during MongoDB processing: {e}")


if __name__ == "__main__":
    log_message("Starting MongoDB skills embedding process")
    process_mongodb_skills()