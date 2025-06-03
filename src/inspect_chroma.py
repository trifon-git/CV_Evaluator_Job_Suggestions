# File: src/inspect_chroma.py

import os
import json
from dotenv import load_dotenv
from chromadb import HttpClient, Settings as ChromaSettings
import numpy as np # Only needed if you want to inspect embeddings themselves

# Load environment variables from .env file at the project root
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir)
    dotenv_path = os.path.join(project_root_dir, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print(f"INFO (inspect_chroma): .env loaded from {dotenv_path}")
    else:
        print(f"WARNING (inspect_chroma): .env file not found at {dotenv_path}.")
except Exception as e:
    print(f"ERROR (inspect_chroma) loading .env: {e}")

# ChromaDB Configuration (ensure these match your .env)
CHROMA_HOST = os.getenv('CHROMA_HOST')
CHROMA_PORT_STR = os.getenv('CHROMA_PORT')
COLLECTION_NAME = os.getenv('CHROMA_COLLECTION') # The name of your collection, e.g., 'job_embeddings'

# How many items to fetch and display
LIMIT_RESULTS = 100 # Adjust as needed

def inspect_collection():
    if not all([CHROMA_HOST, CHROMA_PORT_STR, COLLECTION_NAME]):
        print("ERROR: ChromaDB connection details (CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION) not fully set in .env.")
        return

    try:
        CHROMA_PORT = int(CHROMA_PORT_STR)
    except (ValueError, TypeError):
        print(f"ERROR: CHROMA_PORT ('{CHROMA_PORT_STR}') in .env is not a valid integer. Defaulting to 8000 for connection attempt.")
        CHROMA_PORT = 8000

    print(f"\n--- Attempting to connect to ChromaDB ---")
    print(f"Host: {CHROMA_HOST}, Port: {CHROMA_PORT}, Collection: {COLLECTION_NAME}")

    try:
        client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=ChromaSettings(anonymized_telemetry=False))
        print("Successfully created ChromaDB client.")

        # Check if collection exists
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            print(f"Successfully connected to collection: '{collection.name}'")
            print(f"Collection ID: {collection.id}")
            print(f"Collection item count: {collection.count()}")
            print(f"Collection metadata (distance metric, etc.): {collection.metadata}")

            if collection.count() == 0:
                print("Collection is empty.")
                return

            print(f"\n--- Fetching up to {LIMIT_RESULTS} items from collection '{COLLECTION_NAME}' ---")
            results = collection.get(
                limit=LIMIT_RESULTS,
                include=["metadatas", "documents"] # "documents" usually stores the text that was embedded
                                                  # "embeddings" can also be included but will be very long
            )

            if results and results.get('ids'):
                print(f"Found {len(results['ids'])} items (limited to {LIMIT_RESULTS}):")
                for i, doc_id in enumerate(results['ids']):
                    print(f"\n--- Item {i+1} ---")
                    print(f"ID (MongoDB _id): {doc_id}")
                    
                    metadata = results['metadatas'][i] if results['metadatas'] and i < len(results['metadatas']) else {}
                    print(f"Metadata: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
                    
                    document_text = results['documents'][i] if results['documents'] and i < len(results['documents']) else "N/A"
                    print(f"Document (text embedded by [STEP4]): {document_text[:300]}{'...' if len(document_text) > 300 else ''}") # Print snippet
                    
                    # To inspect embeddings (will be very long):
                    # if "embeddings" in results and results["embeddings"] and i < len(results["embeddings"]):
                    #     embedding_snippet = np.array(results["embeddings"][i][:5]) # First 5 dimensions
                    #     print(f"Embedding (first 5 dims): {embedding_snippet} ...")
            else:
                print("No items found in the collection or results format unexpected.")

        except Exception as e_coll: # Handles if collection does not exist or other errors
            print(f"ERROR accessing collection '{COLLECTION_NAME}': {e_coll}")
            print("Available collections:")
            try:
                collections = client.list_collections()
                if collections:
                    for coll in collections:
                        print(f"  - {coll.name} (ID: {coll.id}, Count: {coll.count()})")
                else:
                    print("  (No collections found on server)")
            except Exception as e_list:
                print(f"  Could not list collections: {e_list}")


    except Exception as e:
        print(f"ERROR connecting to ChromaDB or during inspection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_collection()
