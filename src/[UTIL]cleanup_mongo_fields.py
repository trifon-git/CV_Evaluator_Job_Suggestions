import os
from pymongo import MongoClient
from dotenv import load_dotenv
import sys

# --- Configuration --- 
def load_env_vars():
    """Loads environment variables from .env file and returns them."""
    # Determine project root to find .env file
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir) # Assuming src is one level down from project root
    dotenv_path = os.path.join(project_root_dir, '.env')

    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path, override=True)
        print(f"INFO (cleanup_mongo_fields): .env file loaded from {dotenv_path}")
    else:
        print(f"WARNING (cleanup_mongo_fields): .env file NOT found at {dotenv_path}. Relying on system environment variables.")

    mongo_uri = os.getenv('MONGO_URI')
    mongo_db_name = os.getenv('MONGO_DB_NAME')
    mongo_collection_name = os.getenv('MONGO_COLLECTION')

    if not all([mongo_uri, mongo_db_name, mongo_collection_name]):
        print("ERROR (cleanup_mongo_fields): MONGO_URI, MONGO_DB_NAME, and MONGO_COLLECTION must be set in the environment.")
        sys.exit(1)
    
    return mongo_uri, mongo_db_name, mongo_collection_name

def cleanup_mongo_fields():
    """Connects to MongoDB, finds documents without 'skills' and removes other specified fields."""
    mongo_uri, mongo_db_name, mongo_collection_name = load_env_vars()

    print(f"INFO (cleanup_mongo_fields): Attempting to connect to MongoDB at '{mongo_uri[:20]}...' Database: '{mongo_db_name}', Collection: '{mongo_collection_name}'")

    client = None
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping') # Verify connection
        print("INFO (cleanup_mongo_fields): Successfully connected to MongoDB.")
        
        db = client[mongo_db_name]
        collection = db[mongo_collection_name]

        # Query for documents where 'skills' does not exist OR 'skills' is an empty array
        query = {
            "$or": [
                {"skills": {"$exists": False}},
                {"skills": []} # Matches empty array
            ]
        }

        fields_to_unset = {
            "experience_level_required": "",
            "language_requirements": "",
            "education_level_preferred": "",
            "job_type": ""
        }

        # Find documents matching the query
        documents_to_update = list(collection.find(query, {"_id": 1}))
        num_docs_to_update = len(documents_to_update)

        if num_docs_to_update == 0:
            print("INFO (cleanup_mongo_fields): No documents found matching the criteria (missing or empty 'skills' field).")
            return

        print(f"INFO (cleanup_mongo_fields): Found {num_docs_to_update} documents to process.")

        # Perform the update operation to unset the fields
        # We can do this in bulk for efficiency if the criteria for unsetting is the same for all matched docs
        # However, to be safe and log each, we can iterate. For larger datasets, bulk is better.
        
        # For this script, let's use update_many for efficiency
        update_result = collection.update_many(
            query, 
            {"$unset": fields_to_unset}
        )

        updated_count = update_result.modified_count
        matched_count = update_result.matched_count

        print(f"INFO (cleanup_mongo_fields): MongoDB update operation summary:")
        print(f"  Documents matched by query: {matched_count}")
        print(f"  Documents modified (fields unset): {updated_count}")

        if updated_count < num_docs_to_update:
            print(f"WARNING (cleanup_mongo_fields): {num_docs_to_update - updated_count} documents were matched but not modified. This might be unexpected.")

    except Exception as e:
        print(f"ERROR (cleanup_mongo_fields): An error occurred: {e}")
    finally:
        if client:
            client.close()
            print("INFO (cleanup_mongo_fields): MongoDB connection closed.")

if __name__ == "__main__":
    print("--- cleanup_mongo_fields.py: Script Started ---")
    cleanup_mongo_fields()
    print("--- cleanup_mongo_fields.py: Script Finished ---")