import os
from dotenv import load_dotenv
from chromadb import HttpClient
import time

# Load environment variables
load_dotenv()

def log_message(message):
    """Print a timestamped log message"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {message}")

def reset_chroma_collection():
    """Reset (delete and recreate) the ChromaDB collection specified in .env"""
    # Get configuration from environment variables
    CHROMA_HOST = os.getenv('CHROMA_HOST')
    CHROMA_PORT = int(os.getenv('CHROMA_PORT', '8000'))
    COLLECTION_NAME = os.getenv('CHROMA_COLLECTION')
    
    if not all([CHROMA_HOST, CHROMA_PORT, COLLECTION_NAME]):
        log_message("ERROR: Missing required environment variables. Please check your .env file.")
        log_message(f"CHROMA_HOST: {'Set' if CHROMA_HOST else 'Missing'}")
        log_message(f"CHROMA_PORT: {'Set' if CHROMA_PORT else 'Missing'}")
        log_message(f"COLLECTION_NAME: {'Set' if COLLECTION_NAME else 'Missing'}")
        return False
    
    log_message(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}")
    
    try:
        # Connect to ChromaDB
        chroma_client = HttpClient(
            host=CHROMA_HOST, 
            port=CHROMA_PORT,
            ssl=False,
            headers={"accept": "application/json", "Content-Type": "application/json"}
        )
        
        # Check if collection exists
        collections = chroma_client.list_collections()
        collection_exists = any(collection.name == COLLECTION_NAME for collection in collections)
        
        if collection_exists:
            log_message(f"Found existing collection: {COLLECTION_NAME}")
            
            # Get collection stats before deletion
            try:
                collection = chroma_client.get_collection(COLLECTION_NAME)
                collection_info = collection.get()
                item_count = len(collection_info.get('ids', []))
                log_message(f"Collection contains {item_count} items")
            except Exception as e:
                log_message(f"Could not get collection stats: {e}")
                item_count = "unknown number of"
            
            # Delete the collection
            log_message(f"Deleting collection: {COLLECTION_NAME}")
            chroma_client.delete_collection(COLLECTION_NAME)
            log_message(f"Successfully deleted collection with {item_count} items")
            
            # Small delay to ensure deletion is processed
            time.sleep(1)
        else:
            log_message(f"Collection {COLLECTION_NAME} does not exist yet")
        
        # Create a new empty collection
        log_message(f"Creating new empty collection: {COLLECTION_NAME}")
        chroma_client.create_collection(COLLECTION_NAME)
        log_message(f"Successfully created empty collection: {COLLECTION_NAME}")
        
        return True
        
    except Exception as e:
        log_message(f"ERROR: Failed to reset ChromaDB collection: {e}")
        return False

if __name__ == "__main__":
    log_message("Starting ChromaDB collection reset process")
    
    # Ask for confirmation
    print("\n⚠️  WARNING: This will delete ALL data in your ChromaDB collection! ⚠️")
    confirmation = input("Type 'YES' to confirm deletion: ")
    
    if confirmation.strip().upper() == "YES":
        success = reset_chroma_collection()
        if success:
            log_message("✅ ChromaDB collection reset completed successfully")
        else:
            log_message("❌ ChromaDB collection reset failed")
    else:
        log_message("Operation cancelled by user")