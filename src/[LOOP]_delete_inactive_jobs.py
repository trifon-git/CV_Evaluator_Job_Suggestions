import os
from pymongo import MongoClient
from chromadb import HttpClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB config
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('MONGO_DB_NAME')
COLLECTION_NAME = os.getenv('MONGO_COLLECTION')

# ChromaDB config
CHROMA_HOST = os.getenv('CHROMA_HOST')
CHROMA_PORT = int(os.getenv('CHROMA_PORT'))
CHROMA_COLLECTION = os.getenv('CHROMA_COLLECTION')

def delete_inactive_from_mongo():
    print("Connecting to MongoDB...")
    mongo_client = MongoClient(MONGO_URI)
    mongo_db = mongo_client[DB_NAME]
    mongo_collection = mongo_db[COLLECTION_NAME]
    print("Searching for inactive jobs in MongoDB...")
    inactive_jobs = list(mongo_collection.find({"Status": "inactive"}, {"_id": 1}))
    if not inactive_jobs:
        print("✅ No inactive jobs found in MongoDB.")
        mongo_client.close()
        return
    print(f"Found {len(inactive_jobs)} inactive jobs in MongoDB. Deleting...")
    result = mongo_collection.delete_many({"Status": "inactive"})
    print(f"✅ Deleted {result.deleted_count} inactive jobs from MongoDB.")
    mongo_client.close()

def delete_inactive_from_chroma():
    print("Connecting to ChromaDB...")
    chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    chroma_collection = chroma_client.get_collection(CHROMA_COLLECTION)
    print("Fetching all entries from ChromaDB...")
    results = chroma_collection.get(include=['metadatas'], limit=None)
    ids_to_delete = []
    for id, metadata in zip(results['ids'], results['metadatas']):
        if metadata.get("Status") == "inactive":
            ids_to_delete.append(id)
    if not ids_to_delete:
        print("✅ No inactive jobs found in ChromaDB.")
        return
    print(f"Found {len(ids_to_delete)} inactive jobs in ChromaDB. Deleting in batches...")
    batch_size = 100
    deleted_chroma = 0
    for i in range(0, len(ids_to_delete), batch_size):
        batch_ids = ids_to_delete[i:i+batch_size]
        try:
            chroma_collection.delete(ids=batch_ids)
            deleted_chroma += len(batch_ids)
            print(f"Deleted batch {i//batch_size+1}: {len(batch_ids)} jobs (Total deleted: {deleted_chroma})")
        except Exception as e:
            print(f"❌ Error deleting from ChromaDB for batch {i//batch_size+1}: {e}")
    print(f"✅ Deleted {deleted_chroma} inactive jobs from ChromaDB.")

if __name__ == "__main__":
    # Uncomment the operation you want to perform:
    # To delete from Mongo only:
    # delete_inactive_from_mongo()
    # To delete from Chroma only:
    # delete_inactive_from_chroma()
    # To do both:
    delete_inactive_from_mongo()
    delete_inactive_from_chroma()