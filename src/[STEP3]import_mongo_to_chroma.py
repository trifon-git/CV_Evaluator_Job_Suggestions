from pymongo import MongoClient
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# === CONFIG ===
MONGO_URI = os.getenv('MONGO_URI')
CHROMA_HOST = os.getenv('CHROMA_HOST')
CHROMA_PORT = int(os.getenv('CHROMA_PORT'))
MODEL_NAME = os.getenv('MODEL_NAME')
COLLECTION_NAME = os.getenv('CHROMA_COLLECTION')
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
 
# === CHUNKING FUNCTION ===
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

# === CONNECT TO MONGODB AND CHROMADB ===
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client["job_scraper"]
mongo_collection = mongo_db["jobs"]

chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# === LOAD EMBEDDING MODEL ===
model = SentenceTransformer(MODEL_NAME)

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
            "Category": job["Category"]
        })

    if not ids:
        continue

    try:
        embeddings = model.encode(texts, normalize_embeddings=True)
        chroma_collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
    except Exception as e:
        print(f"❌ Failed to add batch: {e}")

print("✅ Done syncing new jobs into ChromaDB")