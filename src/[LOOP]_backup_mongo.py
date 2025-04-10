from datetime import datetime, timedelta
from pymongo import MongoClient
from tqdm import tqdm
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# === CONFIG ===
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('MONGO_DB_NAME')
COLLECTION_NAME = os.getenv('MONGO_COLLECTION')
BACKUP_COLLECTION_PREFIX = "backup"

# Backup retention configuration
DAILY_RETENTION = int(os.getenv('DAILY_RETENTION', '7'))
WEEKLY_RETENTION = int(os.getenv('WEEKLY_RETENTION', '4'))
MONTHLY_RETENTION = int(os.getenv('MONTHLY_RETENTION', '6'))

def create_backup(backup_type="daily"):
    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    source_collection = db[COLLECTION_NAME]

    # Create backup collection name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_collection_name = f"{BACKUP_COLLECTION_PREFIX}_{backup_type}_{timestamp}"
    backup_collection = db[backup_collection_name]

    try:
        # Get total document count for progress bar
        total_docs = source_collection.count_documents({})
        print(f"Found {total_docs} documents to backup")

        # Copy documents with progress bar
        docs = []
        for doc in tqdm(source_collection.find({}), 
                       total=total_docs,
                       desc="Reading documents",
                       unit="docs"):
            docs.append(doc)

        if docs:
            # Show progress for insertion
            print("Inserting documents into backup collection...")
            with tqdm(total=len(docs), desc="Writing backup", unit="docs") as pbar:
                backup_collection.insert_many(docs)
                pbar.update(len(docs))
            
            print(f"✅ {backup_type.capitalize()} backup completed: {backup_collection_name}")
            print(f"📊 Documents backed up: {len(docs)}")
        else:
            print("No documents found to backup")

        # Rotate old backups
        rotate_backups(db, backup_type)

    except Exception as e:
        print(f"❌ Error during backup: {e}")
        if backup_collection_name in db.list_collection_names():
            print(f"Cleaning up failed backup collection: {backup_collection_name}")
            db[backup_collection_name].drop()
    finally:
        client.close()

def rotate_backups(db, backup_type):
    retention = {
        "daily": DAILY_RETENTION,
        "weekly": WEEKLY_RETENTION,
        "monthly": MONTHLY_RETENTION
    }

    print("\nChecking for old backups to rotate...")
    backup_pattern = f"{BACKUP_COLLECTION_PREFIX}_{backup_type}_"
    backup_collections = sorted([
        coll for coll in db.list_collection_names()
        if coll.startswith(backup_pattern)
    ])

    if len(backup_collections) > retention[backup_type]:
        to_remove = len(backup_collections) - retention[backup_type]
        print(f"Found {to_remove} old backup(s) to remove")
        
        for collection in tqdm(backup_collections[:to_remove], 
                             desc="Rotating old backups",
                             unit="collections"):
            db[collection].drop()
            print(f"🗑️ Removed old backup: {collection}")
    else:
        print("No backup rotation needed")

if __name__ == "__main__":
    # You can run this script with different backup types
    backup_type = "daily"  # or "weekly" or "monthly"
    create_backup(backup_type)