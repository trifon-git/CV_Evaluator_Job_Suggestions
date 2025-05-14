from pymongo import MongoClient
from chromadb import HttpClient
from chromadb.config import Settings
from tqdm import tqdm
from dotenv import load_dotenv
import os
import requests
import numpy as np
import time
import urllib3

# Disable SSL warnings if VERIFY_SSL is set to false
if os.getenv('VERIFY_SSL', 'true').lower() != 'true':
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

# === CONFIG ===
MONGO_URI = os.getenv('MONGO_URI')
CHROMA_HOST = os.getenv('CHROMA_HOST')
CHROMA_PORT = int(os.getenv('CHROMA_PORT'))
COLLECTION_NAME = os.getenv('CHROMA_COLLECTION')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10'))
# Remote embedding API configuration
EMBEDDING_API_URL = os.getenv('EMBEDDING_API_URL')
VERIFY_SSL = os.getenv('VERIFY_SSL', 'true').lower() == 'true'
 
# === CHUNKING FUNCTION ===
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

# === REMOTE EMBEDDING FUNCTION ===
def get_remote_embeddings(texts):
    """Call remote API to get embeddings for texts."""
    try:
        print(f"Calling remote embedding API at {EMBEDDING_API_URL}")
        response = requests.post(EMBEDDING_API_URL, json={"texts": texts}, verify=VERIFY_SSL)
        response.raise_for_status()
        embeddings = response.json().get("embeddings", [])
        if not embeddings:
            print("Warning: Empty embedding response from API")
            return []
        return embeddings
    except Exception as e:
        print(f"Error calling embedding API: {e}")
        return []

# === CONNECT TO MONGODB AND CHROMADB ===
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client["job_scraper"]
mongo_collection = mongo_db["jobs"]

# Update ChromaDB client initialization to use v2 API
chroma_client = HttpClient(
    host=CHROMA_HOST, 
    port=CHROMA_PORT,
    ssl=False,
    headers={"accept": "application/json", "Content-Type": "application/json"}
)
chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# === GET EXISTING IDS FROM CHROMADB ===
existing_ids = set(chroma_collection.get()['ids'])
print(f"Found {len(existing_ids)} existing embeddings in ChromaDB")

# === FETCH JOBS FROM MONGODB ===
# Modified to look for jobs with Skills field instead of html_content
jobs_cursor = mongo_collection.find({
    "Skills": {"$exists": True, "$ne": []},  # Jobs must have non-empty Skills array
    "Status": "active"  # Only process active jobs
})
jobs = [job for job in jobs_cursor if str(job.get("_id")) not in existing_ids]
print(f"Found {len(jobs)} new active jobs with skills to process")

# === PROCESS AND EMBED ===
for batch in tqdm(chunk_list(jobs, BATCH_SIZE), total=len(jobs) // BATCH_SIZE + 1, desc="Embedding new jobs"):
    ids = []
    texts = []
    metadatas = []  # Initialize empty metadatas list

    for job in batch:
        job_id = str(job.get("_id"))
        skills = job.get("Skills", [])

        # Skip jobs with no skills
        if not skills:
            continue

        # Convert skills list to a string for embedding
        skills_text = ", ".join(skills)
        
        # Skip if skills text is too short
        if len(skills_text) < 3:
            continue

        ids.append(job_id)
        texts.append(skills_text)  # Use skills text instead of HTML content
        metadatas.append({
            "Title": job["Title"],
            "URL": job["URL"],
            "Application_URL": job["Application_URL"],
            "Created_At": str(job["Created_At"]),
            "Published_Date": str(job["Published_Date"]) if job.get("Published_Date") else None,
            "Company": job["Company"],
            "Area": job["Area"],
            "Category": job["Category"],
            "Status": job.get("Status", "active"),  # Include Status field with default "active"
            "Skills": skills  # Include the skills list in metadata
        })

    if not ids:
        continue

    try:
        # Use remote embedding API instead of local model
        embeddings = get_remote_embeddings(texts)
        
        if not embeddings:
            print("❌ No embeddings returned from API for this batch")
            continue
            
        # Ensure embeddings are properly formatted
        if len(embeddings) != len(ids):
            print(f"❌ Mismatch: got {len(embeddings)} embeddings for {len(ids)} texts")
            continue
            
        chroma_collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        print(f"✓ Added {len(ids)} jobs to ChromaDB")
        
        # Add a small delay to avoid overwhelming the API
        time.sleep(0.5)
        
    except Exception as e:
        print(f"❌ Failed to add batch: {e}")

print("✅ Done syncing new jobs with skills into ChromaDB")