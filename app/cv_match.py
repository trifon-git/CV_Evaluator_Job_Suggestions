import numpy as np
import time
import os
import traceback
import chromadb
from dotenv import load_dotenv
from chromadb import HttpClient
import requests
import urllib3

# --- Optional Imports ---
try:
    from scipy.spatial.distance import cosine as scipy_cosine
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# --- Configuration ---
load_dotenv()
TOP_N_RESULTS = int(os.getenv('TOP_N_RESULTS', '10'))
CHROMA_HOST = os.getenv('CHROMA_HOST')
CHROMA_PORT_STR = os.getenv('CHROMA_PORT')
COLLECTION_NAME = os.getenv('CHROMA_COLLECTION')
EMBEDDING_API_URL = os.getenv('EMBEDDING_API_URL', '')
VERIFY_SSL_STR = os.getenv('VERIFY_SSL', 'true')

try: CHROMA_PORT = int(CHROMA_PORT_STR) if CHROMA_PORT_STR else 8000
except ValueError: print(f"Warning: Invalid CHROMA_PORT '{CHROMA_PORT_STR}', using 8000."); CHROMA_PORT = 8000
VERIFY_SSL = VERIFY_SSL_STR.lower() == 'true'
if not VERIFY_SSL: urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# NLTK download function removed

# --- Core Functions ---
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    if vec1 is None or vec2 is None or not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray): return 0.0
    if vec1.shape != vec2.shape: return 0.0
    if HAS_SCIPY:
        try: sim = 1 - scipy_cosine(vec1, vec2); return sim if not np.isnan(sim) else 0.0
        except Exception: pass # Fall through
    norm1 = np.linalg.norm(vec1); norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: return 0.0
    dot = np.dot(vec1, vec2); sim = dot / (norm1 * norm2)
    return np.clip(sim, -1.0, 1.0)

def get_remote_embedding(texts):
    """Call remote API to get embeddings for texts."""
    if not EMBEDDING_API_URL: print("Error: EMBEDDING_API_URL not configured."); return []
    try:
        response = requests.post(EMBEDDING_API_URL, json={"texts": texts}, verify=VERIFY_SSL)
        response.raise_for_status()
        return response.json().get("embeddings", [])
    except Exception as e: print(f"Error calling embedding API: {e}"); return []

def generate_cv_embedding_remote(cv_text):
    """Generate embeddings for CV text using remote API with chunking."""
    if not cv_text or not isinstance(cv_text, str): return None
    try:
        chunk_size, overlap, chunks = 1000, 200, []
        start = 0
        while start < len(cv_text):
            end = start + chunk_size; chunk = cv_text[start:end]
            if chunk.strip(): chunks.append(chunk)
            next_start = end - overlap
            if next_start <= start: next_start = start + 1
            start = next_start
            if start >= len(cv_text): break
        if not chunks: print("Warning: No text chunks for embedding."); return None
        chunk_embeddings = []
        batch_size = 16
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_embeddings = get_remote_embedding(batch_chunks)
            if batch_embeddings and len(batch_embeddings) == len(batch_chunks): chunk_embeddings.extend(batch_embeddings)
            else:
                for chunk in batch_chunks: # Fallback
                    embedding_response = get_remote_embedding([chunk])
                    if embedding_response: chunk_embeddings.append(embedding_response[0])
                    else: print("⚠️ Skipped a chunk (empty embedding response)")
        if not chunk_embeddings: print("Error: No valid chunks embedded"); return None
        avg_embedding = np.mean(chunk_embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0: avg_embedding = avg_embedding / norm
        return avg_embedding
    except Exception as e: print(f"Error generating remote embedding: {e}"); traceback.print_exc(); return None

# explain_match function removed

# *** Original find_similar_jobs function signature (no filters) ***
def find_similar_jobs(cv_text, top_n=None, active_only=True):
    """Find jobs similar to the provided CV text using remote embedding."""
    if top_n is None: top_n = TOP_N_RESULTS
    cv_embedding = generate_cv_embedding_remote(cv_text)
    if cv_embedding is None: return [], "Error: Failed to generate CV embedding"

    try:
        if not CHROMA_HOST or not CHROMA_PORT or not COLLECTION_NAME:
             return [], "Error: ChromaDB connection details missing"

        chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, ssl=False, headers={"accept": "application/json"})
        try: collection = chroma_client.get_collection(COLLECTION_NAME)
        except Exception as conn_err: return [], f"Error: Connect/find ChromaDB collection '{COLLECTION_NAME}' failed: {conn_err}"

        # --- Build Simple Where Clause ---
        where_clause = {"Status": "active"} if active_only else None
        # ------------------------------------

        print(f"DEBUG: ChromaDB where clause: {where_clause}") # Debug print

        results = collection.query(
            query_embeddings=[cv_embedding.tolist()], n_results=top_n,
            include=["metadatas", "distances"], # No documents or ids needed in include
            where=where_clause
        )

        matches = []
        if results and results.get('ids') and results['ids'] and results['ids'][0]:
            result_ids = results['ids'][0]; len_ids = len(result_ids)
            result_metadatas = results.get('metadatas', [[]])[0]
            if len(result_metadatas) != len_ids: result_metadatas = ([{}] * len_ids)
            result_distances = results.get('distances', [[]])[0]
            if len(result_distances) != len_ids: result_distances = ([0.0] * len_ids)

            for chroma_id, metadata, distance in zip(result_ids, result_metadatas, result_distances):
                 if not isinstance(metadata, dict): metadata = {}; print(f"Warning: Bad metadata format for ID {chroma_id}")
                 similarity_score = np.exp(-distance) * 100 if distance is not None else 0.0
                 matches.append({
                    "chroma_id": chroma_id, "score": similarity_score, "type": "ChromaDB similarity",
                    "Title": metadata.get('Title', 'N/A'), "Company": metadata.get('Company', 'N/A'),
                    "Area": metadata.get('Area', 'N/A'), "url": metadata.get('Application_URL', metadata.get('URL', '#')),
                    "posting_date": metadata.get('Published_Date', 'N/A'),
                    "Status": metadata.get('Status', 'unknown')
                    # No "content" field
                 })
        else: print("No results returned from ChromaDB query or results format unexpected.")
        return matches, "ChromaDB Vector Search"

    except Exception as e: print(f"Error during ChromaDB search: {e}"); traceback.print_exc(); return [], "Error: ChromaDB search failed"

# --- Example Usage Block ---
if __name__ == "__main__":
    print("cv_match.py loaded (simplified - no filtering).")
    # ... (rest of __main__ block remains the same, maybe remove filtered test) ...
    print(f"\n--- find_similar_jobs Test ---")
    if EMBEDDING_API_URL and CHROMA_HOST and CHROMA_PORT and COLLECTION_NAME:
        try:
            sample_cv = "Jeg er en erfaren software udvikler med kompetencer i Python og cloud infrastruktur."
            matches, message = find_similar_jobs(sample_cv, top_n=3)
            print(f"Search Result Message: {message}")
            if matches: print(f"Found {len(matches)} matches.")
            else: print("No matches found by search.")
        except Exception as e: print(f"Error running test search: {e}")
    else: print("Skipping find_similar_jobs test - Config missing.")
