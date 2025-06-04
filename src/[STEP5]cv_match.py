import os
import json
import sys
import numpy as np
import time
from dotenv import load_dotenv
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from chromadb import HttpClient
import requests
import urllib3

# Import the CV skills extraction functionality
from extract_skills_from_cv_file import get_extracted_skills_from_file

# Load environment variables
load_dotenv()

MODEL_NAME_FOR_EMBEDDING = os.getenv('MODEL_NAME', 'paraphrase-multilingual-mpnet-base-v2')
TOP_N_MATCHES_TO_SHOW = 5  
EXPLAIN_TOP_N_CONTRIBUTING_SKILLS = 3  # How many contributing CV skills to show
CHROMA_HOST = os.getenv('CHROMA_HOST')
CHROMA_PORT = int(os.getenv('CHROMA_PORT', '8000'))
COLLECTION_NAME = os.getenv('CHROMA_COLLECTION')
# Remote embedding API configuration
EMBEDDING_API_URL = os.getenv('EMBEDDING_API_URL', '')
VERIFY_SSL = os.getenv('VERIFY_SSL', 'true').lower() == 'true'

# Disable SSL warnings if VERIFY_SSL is set to false
if os.getenv('VERIFY_SSL', 'true').lower() != 'true':
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Try to use scipy for faster cosine calculations if available
try:
    from scipy.spatial.distance import cosine as scipy_cosine
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Using NumPy for cosine similarity calculations.")
    print("Install scipy for better performance: pip install scipy")

# Check if the embedding API is available
def is_embedding_api_available():
    """Check if the embedding API is available by making a test request"""
    if not EMBEDDING_API_URL:
        log_message("No EMBEDDING_API_URL configured")
        return False
        
    try:
        # Make a minimal test request
        test_text = ["test"]
        response = requests.post(
            EMBEDDING_API_URL, 
            json={"texts": test_text}, 
            verify=VERIFY_SSL,
            timeout=5  # Short timeout for quick check
        )
        
        if response.status_code == 200 and response.json().get("embeddings"):
            log_message("✅ Embedding API is available")
            return True
        else:
            log_message(f"⚠️ Embedding API returned unexpected response: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        log_message(f"⚠️ Embedding API is not available: {e}")
        return False
    except Exception as e:
        log_message(f"⚠️ Error checking embedding API: {e}")
        return False

# Initialize the embedding model (lazy loading)
_embedding_model = None
def get_embedding_model():
    """Get the embedding model, loading it if necessary"""
    global _embedding_model
    if _embedding_model is None:
        log_message(f"Loading sentence transformer model: {MODEL_NAME_FOR_EMBEDDING}")
        _embedding_model = SentenceTransformer(MODEL_NAME_FOR_EMBEDDING)
    return _embedding_model

def log_message(msg):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def generate_average_skill_embedding(model, skills_list):
    # Handle edge cases first
    if not skills_list or not isinstance(skills_list, list):
        return None
        
    # Filter out empty/invalid skills
    valid_skills = [s for s in skills_list if isinstance(s, str) and s.strip()]
    if not valid_skills:
        return None
        
    try:
        # Get embeddings for all skills
        skill_embeddings = model.encode(valid_skills, show_progress_bar=False)
        
        if not skill_embeddings.size:
            return None
            
        # Average the embeddings and normalize
        average_embedding = np.mean(skill_embeddings, axis=0)
        norm = np.linalg.norm(average_embedding)
        
        # Avoid division by zero
        return average_embedding / norm if norm > 0 else None
    except Exception as e:
        log_message(f"Failed to generate embeddings for {len(valid_skills)} skills: {e}")
        return None

def cosine_similarity(vec1, vec2):
    # Handle edge cases
    if vec1 is None or vec2 is None:
        return 0.0
        
    # Ensure we're working with numpy arrays
    vec1, vec2 = np.array(vec1), np.array(vec2)
    
    if HAS_SCIPY:
        try:
            similarity = 1 - scipy_cosine(vec1, vec2)
            return similarity if not np.isnan(similarity) else 0.0
        except Exception as e:
            log_message(f"Scipy calculation failed: {e}. Falling back to NumPy.")
    
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm_vec1, norm_vec2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    
    # Avoid division by zero
    return 0.0 if norm_vec1 == 0 or norm_vec2 == 0 else dot_product / (norm_vec1 * norm_vec2)

def get_remote_embedding(texts):
    """Call remote API to get embeddings for texts."""
    if not EMBEDDING_API_URL:
        log_message("Error: EMBEDDING_API_URL not configured.")
        return []
        
    try:
        log_message("Calling remote embedding API...")
        response = requests.post(EMBEDDING_API_URL, json={"texts": texts}, verify=VERIFY_SSL)
        response.raise_for_status()
        embeddings = response.json().get("embeddings", [])
        if not embeddings:
            log_message("Warning: Empty embedding response from API")
            return []
        return embeddings
    except Exception as e:
        log_message(f"Error calling embedding API: {type(e).__name__}")
        return []

def generate_skills_embedding_remote(skills_list):
    """Generate an averaged embedding for a list of skill strings using remote API."""
    if not skills_list:
        log_message("Error: No skills provided to generate embedding.")
        return None

    # Check if API is available, otherwise use local model
    use_api = is_embedding_api_available()
    
    if not use_api:
        log_message("Falling back to local embedding model")
        model = get_embedding_model()
        return generate_average_skill_embedding(model, skills_list)
    
    log_message(f"Generating embedding for {len(skills_list)} skills via remote API...")
    start_time = time.time()

    skill_embeddings = []
    for i, skill in enumerate(skills_list):
        if not skill.strip():
            continue
        log_message(f"Processing skill {i+1}/{len(skills_list)}: '{skill}'")
        embedding_response = get_remote_embedding([skill]) # get_remote_embedding expects a list
        if embedding_response and len(embedding_response) > 0:
            # Assuming embedding_response[0] is the actual embedding vector
            skill_embeddings.append(np.array(embedding_response[0]))
        else:
            log_message(f"⚠️ Skipped skill '{skill}' due to empty embedding response")

    if not skill_embeddings:
        log_message("Error: No valid skill embeddings were generated.")
        return None

    # Average the embeddings
    avg_embedding = np.mean(skill_embeddings, axis=0)
    
    # Normalize
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm
        
    log_message(f"Skills embedding generated in {time.time() - start_time:.2f}s")
    return avg_embedding

def get_job_data_from_chroma(cv_embedding, top_n=None, active_only=True):
    """Get job data from ChromaDB using vector similarity search"""
    if top_n is None:
        top_n = TOP_N_MATCHES_TO_SHOW
        
    job_ads_data = []
    
    try:
        # Check we have the basics
        if not CHROMA_HOST or not CHROMA_PORT or not COLLECTION_NAME:
            log_message("Missing ChromaDB connection details")
            return []
            
        # Connect to the DB
        log_message(f"Connecting to ChromaDB")
        chroma_client = HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            ssl=False,
            headers={"accept": "application/json"}
        )
        
        # Get our collection
        try:
            collection = chroma_client.get_collection(COLLECTION_NAME)
            log_message(f"Connected to collection: {COLLECTION_NAME}")
        except Exception as conn_err:
            log_message(f"Failed to get collection '{COLLECTION_NAME}': {conn_err}")
            return []
            
        # Only show active jobs unless told otherwise
        where_clause = {"Status": "active"} if active_only else None
        if active_only:
            log_message("Filtering for active jobs only")
            
        # Do the search
        log_message("Performing vector search in ChromaDB...")
        results = collection.query(
            query_embeddings=[cv_embedding.tolist()],
            n_results=top_n,
            include=["metadatas", "distances", "documents"],
            where=where_clause
        )
        
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
                
            result_documents = results.get('documents', [[]])[0]
            if len(result_documents) != len_ids:
                result_documents = ([''] * len_ids)
                
            # Build result list
            for chroma_id, metadata, distance, document in zip(result_ids, result_metadatas, result_distances, result_documents):
                # Make sure metadata is valid
                if not isinstance(metadata, dict):
                    metadata = {}
                    log_message(f"Warning: Bad metadata for ID {chroma_id}")
                    
                # Convert distance to similarity score (0-1)
                similarity_score = 1.0 - min(distance, 1.0)
                
                # Extract skills from metadata
                skills_str = metadata.get('skills', '')
                skills_list = [s.strip() for s in skills_str.split(',') if s.strip()] if skills_str else []
                
                # Create a job record similar to the file-based format
                job_record = {
                    "source_details": {
                        "job_title": metadata.get('Title', 'Unknown Job'),
                        "mongo_id": chroma_id,
                        "company": metadata.get('Company', 'N/A'),
                        "area": metadata.get('Area', 'N/A'),
                        "url": metadata.get('URL', '#'),
                        "application_url": metadata.get('Application_URL', '#'),
                        "status": metadata.get('Status', 'unknown')
                    },
                    "extracted_data": {
                        "skills": skills_list
                    },
                    "skill_embedding": None,  # We don't have the actual embedding, but we don't need it
                    "similarity": similarity_score,
                    "distance": distance,
                    "document": document
                }
                
                job_ads_data.append(job_record)
                
            log_message(f"Found {len(job_ads_data)} matching jobs in ChromaDB")
        else:
            log_message("No results from ChromaDB or unexpected format.")
            
        return job_ads_data
        
    except Exception as e:
        log_message(f"ChromaDB search error: {e}")
        import traceback
        traceback.print_exc()
        return []

def explain_job_match(cv_skills, job_skills, cv_embedding, job_embedding):
    """Explain which CV skills contributed most to the match with a job"""
    if not cv_skills or not job_skills or cv_embedding is None or job_embedding is None:
        return []
        
    # Get valid skills for explanation
    valid_cv_skills_for_explain = [s for s in cv_skills if isinstance(s, str) and s.strip()]
    if not valid_cv_skills_for_explain:
        return []
        
    # Get the embedding model
    model = get_embedding_model()
    
    # Generate embeddings for individual CV skills
    cv_individual_skill_vectors = model.encode(valid_cv_skills_for_explain, show_progress_bar=False)
    
    # Calculate similarity of each skill to the job embedding
    skill_contributions = []
    for i, skill in enumerate(valid_cv_skills_for_explain):
        skill_vector = cv_individual_skill_vectors[i]
        similarity = cosine_similarity(skill_vector, job_embedding)
        skill_contributions.append((skill, similarity))
    
    # Sort by contribution (highest similarity first)
    skill_contributions.sort(key=lambda x: x[1], reverse=True)
    
    return skill_contributions[:EXPLAIN_TOP_N_CONTRIBUTING_SKILLS]

# ----------------------
# Main script execution
# ----------------------
if __name__ == "__main__":
    import time  # Add time import for timing operations
    
    log_message("--- Skill Matching Test Started ---")

    # Find the project root and data directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    
    # Get CV file path from command line or use default
    cv_file_path = None
    if len(sys.argv) > 1:
        cv_file_path = sys.argv[1]
        if not os.path.exists(cv_file_path):
            log_message(f"Error: CV file not found at {cv_file_path}")
            sys.exit(1)
    else:
        # Use default CV file from data directory
        default_cv_path = os.path.join(data_dir, "cv_text_example.md")
        if os.path.exists(default_cv_path):
            cv_file_path = default_cv_path
            log_message(f"Using default CV file: {sanitize_path(cv_file_path)}")
        else:
            log_message("No CV file specified and default CV file not found.")
            log_message("Please provide a CV file path as an argument: python test_skill_matching.py path/to/cv_file")
            sys.exit(1)

    # Extract skills from CV file
    log_message(f"Extracting skills from CV file: {sanitize_path(cv_file_path)}")
    cv_skills = get_extracted_skills_from_file(cv_file_path)
    
    if not cv_skills:
        log_message("No skills extracted from CV file or extraction failed. Exiting.")
        sys.exit(1)
    
    log_message(f"Extracted {len(cv_skills)} skills from CV file:")
    log_message(f"CV Skills: {cv_skills}")
    
    # Choose embedding method based on configuration
    use_remote_embedding = EMBEDDING_API_URL and EMBEDDING_API_URL.strip()
    
    if use_remote_embedding:
        log_message("Using remote embedding API for skill vector generation")
        cv_skill_vector = generate_skills_embedding_remote(cv_skills)
    else:
        # Load the embedding model for local embedding
        log_message(f"Using local embedding model: {MODEL_NAME_FOR_EMBEDDING}")
        try:
            embedding_model = SentenceTransformer(MODEL_NAME_FOR_EMBEDDING)
            log_message("Embedding Model loaded.")
            # Generate embedding for CV skills
            cv_skill_vector = generate_average_skill_embedding(embedding_model, cv_skills)
        except Exception as e:
            log_message(f"Failed to load embedding model: {e}")
            sys.exit(1)
    
    if cv_skill_vector is None:
        log_message("Couldn't create embedding for CV skills. Exiting.")
        sys.exit(1)

    # Get matching jobs from ChromaDB
    job_ads = get_job_data_from_chroma(cv_skill_vector, active_only=True)
    
    if not job_ads:
        log_message("No matching jobs found in ChromaDB. Exiting.")
        sys.exit(1)

    # Process match results
    match_results = []
    for job_ad_record in job_ads:
        # Extract job details
        source_details = job_ad_record.get("source_details", {})
        job_title = source_details.get("job_title", "Unknown Job")
        mongo_id = source_details.get("mongo_id", "N/A")
        job_ad_skills = job_ad_record.get("extracted_data", {}).get("skills", [])
        similarity_score = job_ad_record.get("similarity", 0.0)

        # Store match results
        match_results.append({
            "job_title": job_title,
            "mongo_id": mongo_id,
            "similarity": similarity_score,
            "job_skills_count": len(job_ad_skills),
            "job_ad_record_for_explain": job_ad_record  # Store whole record for explainability
        })
    
    # Sort results by similarity
    match_results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Display top matches
    log_message(f"Top {TOP_N_MATCHES_TO_SHOW} matches:")
    if not match_results:
        log_message("No matches found.")
    
    for i, match in enumerate(match_results[:TOP_N_MATCHES_TO_SHOW]):
        # Show basic match info
        log_message(
            f"\n  {i+1}. JOB: \"{match['job_title']}\" (ID: {match['mongo_id']})"
        )
        log_message(
            f"     Overall Skill Profile Similarity: {match['similarity']:.4f} "
            f"(CV Skills: {len(cv_skills)}, Job Skills: {match['job_skills_count']})"
        )

        # --- EXPLAINABILITY ---
        # Show which CV skills contributed most to this match
        job_ad_for_explain = match["job_ad_record_for_explain"]
        job_ad_skills = job_ad_for_explain.get("extracted_data", {}).get("skills", [])
        
        log_message("     Key CV skills contributing to this match:")
        
        # Calculate average job skill vector using get_embedding_model()
        model = get_embedding_model()
        valid_job_skills_for_explain = [s for s in job_ad_skills if isinstance(s, str) and s.strip()]
        job_avg_vector = None
        
        if valid_job_skills_for_explain:
            job_skill_vectors = model.encode(valid_job_skills_for_explain, show_progress_bar=False)
            job_avg_vector = np.mean(job_skill_vectors, axis=0)
            job_avg_vector_norm = np.linalg.norm(job_avg_vector)
            if job_avg_vector_norm > 0:
                job_avg_vector = job_avg_vector / job_avg_vector_norm
                
            # Get contributing skills
            contributing_skills = explain_job_match(cv_skills, job_ad_skills, cv_skill_vector, job_avg_vector)
            
            # Display results
            if contributing_skills:
                for skill, similarity in contributing_skills:
                    log_message(f"       - \"{skill}\" (Sim. to Job Profile: {similarity:.3f})")
            else:
                log_message("       (No individual CV skills strongly aligned with the job's overall profile above threshold 0.4)")
        else:
            log_message("       (No valid job skills to analyze)")

    log_message("\n--- Skill Matching Test Finished ---")
