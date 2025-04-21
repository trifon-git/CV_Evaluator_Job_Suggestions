import numpy as np
import time
import os
import traceback
import chromadb
from dotenv import load_dotenv
from chromadb import HttpClient
import requests
import urllib3

# Load config
load_dotenv()  # get env vars from .env

# Constants 
TOP_N_RESULTS = int(os.getenv('TOP_N_RESULTS', '10'))
CHROMA_HOST = os.getenv('CHROMA_HOST')
CHROMA_PORT_STR = os.getenv('CHROMA_PORT')
COLLECTION_NAME = os.getenv('CHROMA_COLLECTION')
EMBEDDING_API_URL = os.getenv('EMBEDDING_API_URL', '')
VERIFY_SSL_STR = os.getenv('VERIFY_SSL', 'true')

# Fix port config if it's messed up
try:
    CHROMA_PORT = int(CHROMA_PORT_STR) if CHROMA_PORT_STR else 8000
except ValueError:
    print(f"Warning: Can't use CHROMA_PORT '{CHROMA_PORT_STR}', defaulting to 8000.")
    CHROMA_PORT = 8000

# Handle SSL verification
VERIFY_SSL = VERIFY_SSL_STR.lower() == 'true'
if not VERIFY_SSL:
    # Skip the warnings if we're ignoring SSL
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_remote_embedding(texts):
    """Get embeddings from API"""
    if not EMBEDDING_API_URL:
        print("Error: EMBEDDING_API_URL missing.")
        return []

    try:
        # Hit the embedding API
        response = requests.post(
            EMBEDDING_API_URL,
            json={"texts": texts},
            verify=VERIFY_SSL
        )
        response.raise_for_status()
        return response.json().get("embeddings", [])
    except Exception as e:
        print(f"Error with embedding API: {e}")
        return []


def generate_cv_embedding_remote(cv_text):
    """Chunk CV, get embeddings for each chunk, average them"""
    if not cv_text or not isinstance(cv_text, str):
        return None

    try:
        # Settings for chunking
        chunk_size = 1000
        overlap = 200
        chunks = []

        # Create overlapping chunks of text
        start = 0
        while start < len(cv_text):
            end = start + chunk_size
            chunk = cv_text[start:end]
            if chunk.strip():  # Skip empty chunks
                chunks.append(chunk)

            # Next chunk start position
            next_start = end - overlap
            if next_start <= start:  # Make sure we move forward
                next_start = start + 1
            start = next_start

            if start >= len(cv_text):
                break

        if not chunks:
            print("Warning: No chunks to embed.")
            return None

        # Process chunks in batches to avoid API overload
        chunk_embeddings = []
        batch_size = 16

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_embeddings = get_remote_embedding(batch_chunks)

            if batch_embeddings and len(batch_embeddings) == len(batch_chunks):
                chunk_embeddings.extend(batch_embeddings)
            else:
                # Try one at a time if batch failed
                for chunk in batch_chunks:
                    embedding_response = get_remote_embedding([chunk])
                    if embedding_response:
                        chunk_embeddings.append(embedding_response[0])
                    else:
                        print("⚠️ Skipping chunk - got empty response")

        if not chunk_embeddings:
            print("Error: No chunks were embedded")
            return None

        # Average the embeddings
        avg_embedding = np.mean(chunk_embeddings, axis=0)
        
        # Normalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        return avg_embedding

    except Exception as e:
        print(f"Embedding generation failed: {e}")
        traceback.print_exc()
        return None


def find_similar_jobs(cv_text, top_n=None, active_only=True):
    """Find matching jobs using vector search"""
    # Fallback to default if not specified
    if top_n is None:
        top_n = TOP_N_RESULTS

    # Get embedding for the CV
    cv_embedding = generate_cv_embedding_remote(cv_text)
    if cv_embedding is None:
        return [], "Couldn't generate CV embedding"

    try:
        # Check we have the basics
        if not CHROMA_HOST or not CHROMA_PORT or not COLLECTION_NAME:
             return [], "Missing ChromaDB connection details"

        # Connect to the DB
        chroma_client = HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            ssl=False,
            headers={"accept": "application/json"}
        )

        # Get our collection
        try:
            collection = chroma_client.get_collection(COLLECTION_NAME)
        except Exception as conn_err:
            return [], f"Failed to get collection '{COLLECTION_NAME}': {conn_err}"

        # Only show active jobs unless told otherwise
        where_clause = {"Status": "active"} if active_only else None

        # Do the search
        results = collection.query(
            query_embeddings=[cv_embedding.tolist()],
            n_results=top_n,
            include=["metadatas", "distances"],
            where=where_clause
        )

        matches = []

        # Process what we found
        if results and results.get('ids') and results['ids'] and results['ids'][0]:
            result_ids = results['ids'][0]
            len_ids = len(result_ids)

            # Get metadata and distances, handle missing data
            result_metadatas = results.get('metadatas', [[]])[0]
            if len(result_metadatas) != len_ids:
                result_metadatas = ([{}] * len_ids)

            result_distances = results.get('distances', [[]])[0]
            if len(result_distances) != len_ids:
                result_distances = ([0.0] * len_ids)

            # Build result list
            for chroma_id, metadata, distance in zip(result_ids, result_metadatas, result_distances):
                # Make sure metadata is valid
                if not isinstance(metadata, dict):
                    metadata = {}
                    print(f"Warning: Bad metadata for ID {chroma_id}")

                # Convert distance to a score people can understand (0-100)
                similarity_score = np.exp(-distance) * 100 if distance is not None else 0.0

                # Add to matches
                matches.append({
                    "chroma_id": chroma_id,
                    "score": similarity_score,
                    "type": "ChromaDB similarity",
                    "Title": metadata.get('Title', 'N/A'),
                    "Company": metadata.get('Company', 'N/A'),
                    "Area": metadata.get('Area', 'N/A'),
                    "url": metadata.get('Application_URL', metadata.get('URL', '#')),
                    "posting_date": metadata.get('Published_Date', 'N/A'),
                    "Status": metadata.get('Status', 'unknown')
                })
        else:
            print("No results from ChromaDB or weird format.")

        return matches, "ChromaDB Vector Search"

    except Exception as e:
        print(f"ChromaDB search error: {e}")
        traceback.print_exc()
        return [], "ChromaDB search failed"


# For testing
if __name__ == "__main__":
    print("cv_match_hf loaded.")

    # Quick test
    print(f"\n--- Test Run ---")
    if EMBEDDING_API_URL and CHROMA_HOST and CHROMA_PORT and COLLECTION_NAME:
        try:
            # Danish CV sample
            sample_cv = "Jeg er en erfaren software udvikler med kompetencer i Python og cloud infrastruktur."

            # Test it
            matches, message = find_similar_jobs(sample_cv, top_n=3)
            print(f"Search result: {message}")

            if matches:
                print(f"Found {len(matches)} matches.")
                # Uncomment to see details
                # print(matches[0])
            else:
                print("No matches found.")

        except Exception as e:
            print(f"Test failed: {e}")
    else:
        print("Skipping test - missing config.")
