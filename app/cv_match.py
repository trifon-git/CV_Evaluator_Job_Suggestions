import numpy as np
import time
import os
import traceback
from sentence_transformers import SentenceTransformer # Still needed for tokenization in generate_cv_embedding
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from chromadb import HttpClient
import requests
import urllib3

# Try to use scipy for faster cosine calculations if available
try:
    from scipy.spatial.distance import cosine as scipy_cosine
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Load environment variables
load_dotenv()

# Configuration from environment variables
MODEL_NAME = os.getenv('MODEL_NAME', 'paraphrase-multilingual-mpnet-base-v2')
TOP_N_RESULTS = int(os.getenv('TOP_N_RESULTS', '10'))
# CV_FILE_PATH removed as it's handled by the UI now
CHROMA_HOST = os.getenv('CHROMA_HOST')
# Remove or comment out this line since port is included in CHROMA_HOST
# CHROMA_PORT = int(os.getenv('CHROMA_PORT'))
COLLECTION_NAME = os.getenv('CHROMA_COLLECTION')
# Remote embedding API configuration
EMBEDDING_API_URL = os.getenv('EMBEDDING_API_URL', '')
VERIFY_SSL = os.getenv('VERIFY_SSL', 'true').lower() == 'true'

# Disable SSL warnings if VERIFY_SSL is set to false
if not VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    if vec1 is None or vec2 is None or not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
        # print(f"Warning: Invalid vectors for similarity calculation")
        return 0.0

    if vec1.shape != vec2.shape:
        # print(f"Shape mismatch: {vec1.shape} vs {vec2.shape}")
        return 0.0

    if HAS_SCIPY:
        try:
            similarity = 1 - scipy_cosine(vec1, vec2)
            return similarity if not np.isnan(similarity) else 0.0
        except Exception as e:
            # print(f"Scipy calculation failed: {e}. Falling back to NumPy.")
            pass # Fall through to NumPy method

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    dot_product = np.dot(vec1, vec2)
    similarity = dot_product / (norm1 * norm2)

    return np.clip(similarity, -1.0, 1.0)

# This function is kept for potential future use or if local model is needed elsewhere
# but the primary find_similar_jobs now uses the remote version.
def generate_cv_embedding(model, cv_text):
    """Generate embeddings for CV text with chunking for long texts."""
    # print("Generating CV embedding...")
    start_time = time.time()

    if not cv_text or not isinstance(cv_text, str):
        # print("Error: Invalid CV text")
        return None

    try:
        # Lazy load model only if this function is called
        print(f"Loading local model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        print("Local model loaded.")

        tokenizer = model.tokenizer
        max_seq_length = model.max_seq_length
        tokens = tokenizer.encode(cv_text)
        total_tokens = len(tokens)

        # print(f"CV text contains {total_tokens} tokens (model limit: {max_seq_length})")

        if total_tokens <= max_seq_length:
            # print("Text within model limits, encoding directly")
            return model.encode(cv_text, normalize_embeddings=True)

        # print("Text exceeds model limit, using chunking strategy")
        chunk_size = max_seq_length
        overlap = max(50, chunk_size // 4)
        step = chunk_size - overlap

        chunk_embeddings = []
        start_index = 0

        while start_index < total_tokens:
            end_index = min(start_index + chunk_size, total_tokens)
            chunk_tokens = tokens[start_index:end_index]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=True)

            if not chunk_text.strip():
                start_index += step
                continue

            chunk_emb = model.encode(chunk_text, normalize_embeddings=True)
            chunk_embeddings.append(chunk_emb)

            if end_index == total_tokens:
                break

            start_index += step

        if not chunk_embeddings:
            # print("Error: No valid embeddings generated after chunking")
            return None

        # print(f"Generated {len(chunk_embeddings)} embedding chunks")

        avg_embedding = np.mean(chunk_embeddings, axis=0)

        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        # print(f"Embedding generated in {time.time() - start_time:.2f}s")
        return avg_embedding

    except Exception as e:
        print(f"Error generating embedding: {e}")
        traceback.print_exc()
        return None

def get_remote_embedding(texts):
    """Call remote API to get embeddings for texts."""
    if not EMBEDDING_API_URL:
        print("Error: EMBEDDING_API_URL is not configured.")
        return []
    try:
        # print(f"Calling remote embedding API at {EMBEDDING_API_URL}")
        response = requests.post(EMBEDDING_API_URL, json={"texts": texts}, verify=VERIFY_SSL)
        response.raise_for_status()
        embeddings = response.json().get("embeddings", [])
        if not embeddings:
            # print("Warning: Empty embedding response from API")
            return []
        return embeddings
    except Exception as e:
        print(f"Error calling embedding API: {e}")
        traceback.print_exc()
        return []

def generate_cv_embedding_remote(cv_text):
    """Generate embeddings for CV text using remote API with chunking for long texts."""
    # print("Generating CV embedding via remote API...")
    start_time = time.time()

    if not cv_text or not isinstance(cv_text, str):
        # print("Error: Invalid CV text")
        return None

    try:
        # Split text into chunks
        chunk_size = 1000  # characters per chunk
        overlap = 200      # character overlap between chunks

        chunks = []
        start = 0
        while start < len(cv_text):
            end = start + chunk_size
            chunk = cv_text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            # Ensure start moves forward even if end hits len(cv_text) exactly
            next_start = end - overlap
            if next_start <= start: # Avoid infinite loop on very small texts or large overlaps
                next_start = start + 1
            start = next_start
            if start >= len(cv_text):
                break

        # print(f"Split CV into {len(chunks)} chunks for remote embedding")

        chunk_embeddings = []
        # Process chunks in batches to potentially improve API efficiency if supported
        batch_size = 16 # Example batch size, adjust as needed
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            # print(f"Processing chunk batch {i//batch_size + 1}")
            batch_embeddings = get_remote_embedding(batch_chunks)
            if batch_embeddings and len(batch_embeddings) == len(batch_chunks):
                chunk_embeddings.extend(batch_embeddings)
            else:
                print(f"⚠️ Issue embedding batch starting at chunk {i}, trying individually...")
                 # Fallback to individual processing for this batch if batch failed
                for chunk in batch_chunks:
                    embedding_response = get_remote_embedding([chunk])
                    if embedding_response:
                        chunk_embeddings.append(embedding_response[0])
                    else:
                        print("⚠️ Skipped a chunk due to empty embedding response")

        if not chunk_embeddings:
            print("Error: No valid chunks were embedded")
            return None

        # Average the embeddings
        avg_embedding = np.mean(chunk_embeddings, axis=0)

        # Normalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        # print(f"Remote embedding generated in {time.time() - start_time:.2f}s")
        return avg_embedding

    except Exception as e:
        print(f"Error generating remote embedding: {e}")
        traceback.print_exc()
        return None

def find_similar_jobs(cv_text, top_n=None, active_only=True):
    """Find jobs similar to the provided CV text using remote embedding."""
    if top_n is None:
        top_n = TOP_N_RESULTS

    cv_embedding = generate_cv_embedding_remote(cv_text)

    if cv_embedding is None:
        return None, "Error: Failed to generate CV embedding"

    try:
        # Check essential config
        if not CHROMA_HOST or not COLLECTION_NAME:
             return None, "Error: ChromaDB connection details missing in configuration"

        chroma_client = HttpClient(
            host=CHROMA_HOST,
            ssl=False,
            headers={"accept": "application/json", "Content-Type": "application/json"},
            api_version="v2"  # Explicitly use v2 API
        )
        
        # Verify connection and collection existence
        try:
            chroma_client.heartbeat()  # Check if server is reachable
            collection = chroma_client.get_or_create_collection(COLLECTION_NAME)  # Use get_or_create instead
        except Exception as conn_err:
            print(f"ChromaDB connection/collection error: {conn_err}")
            return None, f"Error: Could not connect to or find ChromaDB collection '{COLLECTION_NAME}'"


        search_start = time.time()

        # Add filter for active jobs if requested
        where_filter = {"Status": "active"} if active_only else None
        # if active_only:
            # print("Filtering for active jobs only")

        results = collection.query(
            query_embeddings=[cv_embedding.tolist()],
            n_results=top_n,
            include=["metadatas", "distances", "documents"], # Include documents for potential display
            where=where_filter
        )

        matches = []
        if results and results.get('ids') and results['ids'][0]: # Check if results are valid
            for idx, (metadata, distance, content) in enumerate(zip(
                results['metadatas'][0],
                results['distances'][0],
                results['documents'][0] if results.get('documents') else [''] * len(results['ids'][0]) # Handle missing documents safely
            )):
                # Using exponential decay for more intuitive scoring
                # Distance in ChromaDB is often squared L2, adjust if using cosine directly
                # Assuming distance is squared L2, smaller is better.
                # If distance was cosine similarity (larger is better), calculation needs inversion.
                # Let's stick to the previous exp(-distance) assuming L2 distance.
                similarity_score = np.exp(-distance) * 100
                matches.append({
                    "score": similarity_score,
                    "type": "ChromaDB similarity",
                    "Title": metadata.get('Title', 'Unknown Position'),
                    "Company": metadata.get('Company', 'N/A'),
                    "Area": metadata.get('Area', 'N/A'),
                    "url": metadata.get('Application_URL', metadata.get('URL', '#')), # Fallback URL
                    "posting_date": metadata.get('Published_Date', 'N/A'),
                    "content": content, # Include the document content
                    "Status": metadata.get('Status', 'unknown')
                })
        else:
            print("No results returned from ChromaDB query.")


        # print(f"Search completed in {time.time() - search_start:.2f}s")
        return matches, "ChromaDB Vector Search"

    except Exception as e:
        print(f"Error during ChromaDB search: {e}")
        traceback.print_exc()
        return None, "Error: ChromaDB search failed"

