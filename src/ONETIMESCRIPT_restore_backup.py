from pymongo import MongoClient
from datetime import datetime
import sys
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# === CONFIG ===
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('MONGO_DB_NAME')
COLLECTION_NAME = os.getenv('MONGO_COLLECTION')
BACKUP_PREFIX = "backup"

def init_mongodb():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return client, db

def restore_latest_backup():
    client, db = init_mongodb()
    try:
        # Get all backup collections
        backup_collections = [
            coll for coll in db.list_collection_names()
            if coll.startswith(BACKUP_PREFIX)
        ]
        
        if not backup_collections:
            print("❌ No backup collections found!")
            return False
            
        # Sort to get the most recent backup
        latest_backup = sorted(backup_collections)[-1]
        print(f"📦 Found latest backup: {latest_backup}")
        
        # Confirm with user
        confirm = input(f"⚠️ This will replace the current '{COLLECTION_NAME}' collection with '{latest_backup}'. Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return False
        
        # Rename the current collection to a temporary name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        old_collection_name = f"{COLLECTION_NAME}_old_{timestamp}"
        
        # Rename current collection to backup
        if COLLECTION_NAME in db.list_collection_names():
            db[COLLECTION_NAME].rename(old_collection_name)
            print(f"✅ Current collection backed up as: {old_collection_name}")
        
        # Rename backup to be the current collection
        db[latest_backup].rename(COLLECTION_NAME)
        print(f"✅ Restored {latest_backup} as the active collection")
        
        # Ask if user wants to keep the old collection
        keep_old = input("Keep the old collection backup? (y/n): ")
        if keep_old.lower() != 'y':
            db[old_collection_name].drop()
            print(f"🗑️ Removed old collection: {old_collection_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during restoration: {e}")
        return False
    finally:
        client.close()

if __name__ == "__main__":
    print("🔄 Starting backup restoration process...")
    if restore_latest_backup():
        print("✅ Restoration completed successfully!")
    else:
        print("❌ Restoration failed or was cancelled.")