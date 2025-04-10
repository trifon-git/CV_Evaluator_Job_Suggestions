import numpy as np
import time
import os
import traceback
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from chromadb import HttpClient

# Load environment variables
load_dotenv()

# Try to use scipy for faster cosine calculations if available
try:
    from scipy.spatial.distance import cosine as scipy_cosine
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Using NumPy for cosine similarity calculations.")
    print("Install scipy for better performance: pip install scipy")

# Configuration from environment variables
MODEL_NAME = os.getenv('MODEL_NAME', 'paraphrase-multilingual-mpnet-base-v2')
TOP_N_RESULTS = int(os.getenv('TOP_N_RESULTS', '10'))
CV_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", os.getenv('CV_FILE_PATH'))
CHROMA_HOST = os.getenv('CHROMA_HOST')
CHROMA_PORT = int(os.getenv('CHROMA_PORT'))
COLLECTION_NAME = os.getenv('CHROMA_COLLECTION')

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    if vec1 is None or vec2 is None or not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
        print(f"Warning: Invalid vectors for similarity calculation")
        return 0.0
    
    if vec1.shape != vec2.shape:
        print(f"Shape mismatch: {vec1.shape} vs {vec2.shape}")
        return 0.0

    if HAS_SCIPY:
        try:
            similarity = 1 - scipy_cosine(vec1, vec2)
            return similarity if not np.isnan(similarity) else 0.0
        except Exception as e:
            print(f"Scipy calculation failed: {e}. Falling back to NumPy.")
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    dot_product = np.dot(vec1, vec2)
    similarity = dot_product / (norm1 * norm2)
    
    return np.clip(similarity, -1.0, 1.0)

def generate_cv_embedding(model, cv_text):
    """Generate embeddings for CV text with chunking for long texts."""
    print("Generating CV embedding...")
    start_time = time.time()
    
    if not cv_text or not isinstance(cv_text, str):
        print("Error: Invalid CV text")
        return None
        
    try:
        tokenizer = model.tokenizer
        max_seq_length = model.max_seq_length
        tokens = tokenizer.encode(cv_text)
        total_tokens = len(tokens)
        
        print(f"CV text contains {total_tokens} tokens (model limit: {max_seq_length})")
        
        if total_tokens <= max_seq_length:
            print("Text within model limits, encoding directly")
            return model.encode(cv_text, normalize_embeddings=True)
            
        print("Text exceeds model limit, using chunking strategy")
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
            print("Error: No valid embeddings generated after chunking")
            return None
            
        print(f"Generated {len(chunk_embeddings)} embedding chunks")
        
        avg_embedding = np.mean(chunk_embeddings, axis=0)
        
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
            
        print(f"Embedding generated in {time.time() - start_time:.2f}s")
        return avg_embedding
        
    except Exception as e:
        print(f"Error generating embedding: {e}")
        traceback.print_exc()
        return None

def find_similar_jobs(model, cv_text, top_n=None, active_only=True):
    """Find jobs similar to the provided CV text."""
    if top_n is None:
        top_n = TOP_N_RESULTS
        
    cv_embedding = generate_cv_embedding(model, cv_text)
    if cv_embedding is None:
        return None, "Error: Failed to generate CV embedding"
        
    print(f"CV embedding shape: {cv_embedding.shape}")
    
    print("Connecting to ChromaDB...")
    try:
        chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collection = chroma_client.get_collection(COLLECTION_NAME)
        
        search_start = time.time()
        
        # Add filter for active jobs if requested
        where_filter = {"Status": "active"} if active_only else None
        if active_only:
            print("Filtering for active jobs only")
        
        results = collection.query(
            query_embeddings=[cv_embedding.tolist()],
            n_results=top_n,
            include=["metadatas", "distances", "documents"],
            where=where_filter
        )
        
        matches = []
        for idx, (metadata, distance, content) in enumerate(zip(
            results['metadatas'][0], 
            results['distances'][0],
            results['documents'][0]
        )):
            # Using exponential decay for more intuitive scoring
            similarity_score = np.exp(-distance) * 100  # Will give scores between 0-100
            matches.append({
                "score": similarity_score,
                "type": "ChromaDB similarity",
                "Title": metadata.get('Title', 'Unknown Position'),  
                "Company": metadata.get('Company', 'N/A'),          
                "Area": metadata.get('Area', 'N/A'),               
                "url": metadata.get('Application_URL', '#'),       
                "posting_date": metadata.get('Published_Date', 'N/A'), 
                "content": content,
                "Status": metadata.get('Status', 'unknown')
            })
        
        print(f"Search completed in {time.time() - search_start:.2f}s")
        return matches, "ChromaDB Vector Search"
        
    except Exception as e:
        print(f"Error during ChromaDB search: {e}")
        traceback.print_exc()
        return None, "Error: ChromaDB search failed"

def main():
    print("\n=== Job Matcher ===\n")
    
    print(f"Reading CV from {CV_FILE_PATH}")
    try:
        with open(CV_FILE_PATH, "r", encoding="utf-8") as f:
            cv_text = f.read()
        print(f"CV text loaded ({len(cv_text)} characters)")
    except FileNotFoundError:
        print(f"Error: File not found: {CV_FILE_PATH}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
        
    if not cv_text.strip():
        print("Error: CV file is empty")
        return
        
    print(f"Loading model: {MODEL_NAME}")
    try:
        model = SentenceTransformer(MODEL_NAME)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    print("\nSearching for matching jobs...")
    matches, method = find_similar_jobs(model, cv_text, active_only=True)
    
    print("\n=== Results ===")
    print(f"Search method: {method}")
    
    if not matches or method.startswith("Error"):
        print("No matches found or search failed")
        return
        
    print(f"Found {len(matches)} potential matches:\n")
    for i, job in enumerate(matches):
        print(f"{i+1}. {job.get('Title', 'Unknown Position')}")
        print(f"   Company: {job.get('Company', 'N/A')}")
        print(f"   Location: {job.get('Area', 'N/A')}")
        print(f"   Posted: {job.get('posting_date', 'N/A')}")
        print(f"   Status: {job.get('Status', 'unknown')}")
        print(f"   Match score: {job.get('score', 0):.2f}")
        print(f"   URL: {job.get('url', '#')}")
        print()
        
    print("Job matching complete")

if __name__ == "__main__":
    main()