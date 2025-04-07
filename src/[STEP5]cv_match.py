import sqlite3
import numpy as np
import time
import os
import traceback
from sentence_transformers import SentenceTransformer

# Try to use scipy for faster cosine calculations if available
try:
    from scipy.spatial.distance import cosine as scipy_cosine
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Using NumPy for cosine similarity calculations.")
    print("Install scipy for better performance: pip install scipy")


# Configuration
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "job_listings.db")
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'  # Must match what scraper used
TOP_N_RESULTS = 10
CV_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cv_text_exampleDK.md")


def vector_to_blob(vector):
    """Convert a numpy vector to a binary blob for storage."""
    if vector is None:
        return None
    return np.array(vector, dtype=np.float32).tobytes()


def blob_to_vector(blob):
    """Convert a binary blob back to a numpy vector."""
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    # Validate inputs
    if vec1 is None or vec2 is None or not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
        print(f"Warning: Invalid vectors for similarity calculation")
        return 0.0
    
    if vec1.shape != vec2.shape:
        print(f"Shape mismatch: {vec1.shape} vs {vec2.shape}")
        return 0.0

    # Try scipy first (faster)
    if HAS_SCIPY:
        try:
            similarity = 1 - scipy_cosine(vec1, vec2)
            return similarity if not np.isnan(similarity) else 0.0
        except Exception as e:
            print(f"Scipy calculation failed: {e}. Falling back to NumPy.")
    
    # NumPy fallback implementation
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    dot_product = np.dot(vec1, vec2)
    similarity = dot_product / (norm1 * norm2)
    
    # Ensure value is in valid range
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
        
        # For short text, encode directly
        if total_tokens <= max_seq_length:
            print("Text within model limits, encoding directly")
            return model.encode(cv_text, normalize_embeddings=True)
            
        # For long text, use chunking approach
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
            
            # Skip empty chunks
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
        
        # Average the chunk embeddings
        avg_embedding = np.mean(chunk_embeddings, axis=0)
        
        # Normalize the result
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
            
        print(f"Embedding generated in {time.time() - start_time:.2f}s")
        return avg_embedding
        
    except Exception as e:
        print(f"Error generating embedding: {e}")
        traceback.print_exc()
        return None


def find_similar_jobs(db_path, model, cv_text, top_n=10):
    """Find jobs similar to the provided CV text."""
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return None, "Error: Database not found"
        
    # Generate embedding from CV text
    cv_embedding = generate_cv_embedding(model, cv_text)
    if cv_embedding is None:
        return None, "Error: Failed to generate CV embedding"
        
    print(f"CV embedding shape: {cv_embedding.shape}")
    
    # Connect to database
    print(f"Connecting to database: {db_path}")
    conn = None
    results = []
    search_method = "Unknown"
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Try to use SQLite VSS extension for vector search
        use_vss = False
        try:
            conn.enable_load_extension(True)
            import sqlite_vss
            sqlite_vss.load(conn)
            print("SQLite VSS extension loaded successfully")
            use_vss = True
            search_method = "Vector Search (VSS)"
        except (ImportError, AttributeError) as e:
            print(f"VSS extension not available: {e}")
            print("Falling back to manual similarity calculation")
            search_method = "Manual Calculation"
        except Exception as e:
            print(f"Unexpected error loading VSS: {e}")
            search_method = "Manual Calculation"
            
        search_start = time.time()
            
        if use_vss:
            # Vector search using VSS extension
            print(f"Performing vector search for top {top_n} matches...")
            
            # Check if VSS table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vss_jobs';")
            if cursor.fetchone() is None:
                print("Error: 'vss_jobs' virtual table doesn't exist")
                search_method = "Error: Missing VSS table"
                return None, search_method
            
            cv_blob = vector_to_blob(cv_embedding)
            
            query = """
                SELECT
                    j.rowid AS job_rowid, j.Title, c.name AS Company, a.name AS Area, j.url,
                    v.distance
                FROM vss_jobs v
                JOIN jobs j ON v.rowid = j.rowid
                LEFT JOIN companies c ON j.Company_ID = c.id
                LEFT JOIN areas a ON j.Area_ID = a.id
                WHERE vss_search(v.embedding, ?)
                ORDER BY v.distance ASC
                LIMIT ?;
            """
            
            try:
                rows = cursor.execute(query, (cv_blob, top_n)).fetchall()
                results = [
                    {
                        "score": 1 - row['distance'], 
                        "type": "VSS similarity",
                        **dict(row)
                    } 
                    for row in rows
                ]
                results.sort(key=lambda x: x['score'], reverse=True)
                print(f"Search completed in {time.time() - search_start:.2f}s")
            except sqlite3.Error as e:
                print(f"VSS query error: {e}")
                search_method = "Error: VSS query failed"
                results = []
                
        else:
            # Manual similarity calculation
            print(f"Fetching all job embeddings for manual comparison...")
            
            query = """
                SELECT
                    j.rowid AS job_rowid, j.Title, c.name AS Company, a.name AS Area, j.url,
                    j.embedding
                FROM jobs j
                LEFT JOIN companies c ON j.Company_ID = c.id
                LEFT JOIN areas a ON j.Area_ID = a.id
                WHERE j.embedding IS NOT NULL;
            """
            
            try:
                fetch_start = time.time()
                all_jobs = cursor.execute(query).fetchall()
                print(f"Fetched {len(all_jobs)} job embeddings in {time.time() - fetch_start:.2f}s")
                
                if not all_jobs:
                    print("No jobs with embeddings found")
                    return [], "No jobs found"
                    
                print("Calculating similarities...")
                calc_start = time.time()
                similarities = []
                
                for job in all_jobs:
                    job_embedding = blob_to_vector(job['embedding'])
                    
                    if job_embedding is not None and job_embedding.shape == cv_embedding.shape:
                        sim = cosine_similarity(cv_embedding, job_embedding)
                        similarities.append({
                            "score": sim,
                            "type": "cosine similarity",
                            "job_rowid": job['job_rowid'],
                            "Title": job['Title'],
                            "Company": job['Company'],
                            "Area": job['Area'],
                            "url": job['url']
                        })
                    else:
                        print(f"Skipping job {job['job_rowid']} due to invalid embedding")
                        
                similarities.sort(key=lambda x: x['score'], reverse=True)
                results = similarities[:top_n]
                print(f"Calculations completed in {time.time() - calc_start:.2f}s")
                
            except Exception as e:
                print(f"Error during manual calculation: {e}")
                traceback.print_exc()
                search_method = "Error: Manual calculation failed"
                results = []
                
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        search_method = "Error: Unhandled exception"
        
    finally:
        if conn:
            conn.close()
            print("Database connection closed")
            
    return results, search_method


def main():
    print("\n=== Job Matcher ===\n")
    
    # Read CV text
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
        
    # Load the model
    print(f"Loading model: {MODEL_NAME}")
    try:
        model = SentenceTransformer(MODEL_NAME)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Find matching jobs
    print("\nSearching for matching jobs...")
    matches, method = find_similar_jobs(DB_PATH, model, cv_text, top_n=TOP_N_RESULTS)
    
    # Display results
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
        print(f"   Match score: {job.get('score', 0):.2f}")
        print(f"   URL: {job.get('url', '#')}")
        print()
        
    print("Job matching complete")


if __name__ == "__main__":
    main()