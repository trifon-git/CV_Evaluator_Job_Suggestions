from pymongo import MongoClient
from chromadb import HttpClient
from chromadb.config import Settings
from tqdm import tqdm
from dotenv import load_dotenv
import os
import requests
import numpy as np
import time

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
settings = Settings(
    chroma_api_impl="chromadb.api.fastapi.FastAPI",
    chroma_server_host=CHROMA_HOST,
    chroma_server_http_port=CHROMA_PORT,
    chroma_server_ssl_enabled=False
)
chroma_client = HttpClient(settings=settings)
chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# === GET EXISTING IDS FROM CHROMADB ===
existing_ids = set(chroma_collection.get()['ids'])
print(f"Found {len(existing_ids)} existing embeddings in ChromaDB")

# === FETCH JOBS FROM MONGODB ===
jobs_cursor = mongo_collection.find({"html_content": {"$exists": True, "$ne": None}})
jobs = [job for job in jobs_cursor if str(job.get("_id")) not in existing_ids]
print(f"Found {len(jobs)} new jobs to process")

# === PROCESS AND EMBED ===
for batch in tqdm(chunk_list(jobs, BATCH_SIZE), total=len(jobs) // BATCH_SIZE + 1, desc="Embedding new jobs"):
    ids = []
    texts = []
    metadatas = []  # Initialize empty metadatas list

    for job in batch:
        job_id = str(job.get("_id"))
        html = job.get("html_content")

        if not html or len(html) < 50:
            continue

        ids.append(job_id)
        texts.append(html)
        metadatas.append({
            "Title": job["Title"],
            "URL": job["URL"],
            "Application_URL": job["Application_URL"],
            "Created_At": str(job["Created_At"]),
            "Published_Date": str(job["Published_Date"]) if job["Published_Date"] else None,
            "Company": job["Company"],
            "Area": job["Area"],
            "Category": job["Category"],
            "Status": job.get("Status", "active")  # Include Status field with default "active"
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

print("✅ Done syncing new jobs into ChromaDB")