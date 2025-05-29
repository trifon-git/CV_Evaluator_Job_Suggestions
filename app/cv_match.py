import numpy as np
import time
import os
import traceback
import chromadb
from dotenv import load_dotenv
from chromadb import HttpClient, Settings as ChromaSettings
import requests
import urllib3
import json # Added for parsing Skills_Json_Str from metadata

# Load config
load_dotenv()

# Constants
TOP_N_RESULTS_DEFAULT = int(os.getenv('TOP_N_RESULTS', '20'))
CHROMA_HOST = os.getenv('CHROMA_HOST')
CHROMA_PORT_STR = os.getenv('CHROMA_PORT')
COLLECTION_NAME = os.getenv('CHROMA_COLLECTION')
EMBEDDING_API_URL = os.getenv('EMBEDDING_API_URL', '')
VERIFY_SSL_STR = os.getenv('VERIFY_SSL', 'true')
EXPLAIN_TOP_N_CONTRIBUTING_SKILLS = int(os.getenv('EXPLAIN_TOP_N_CONTRIBUTING_SKILLS', 3))
MODEL_NAME_FOR_EMBEDDING = os.getenv('MODEL_NAME', 'paraphrase-multilingual-mpnet-base-v2')

try:
    CHROMA_PORT = int(CHROMA_PORT_STR) if CHROMA_PORT_STR else 8000
except ValueError:
    print(f"Warning: Invalid CHROMA_PORT '{CHROMA_PORT_STR}', defaulting to 8000.")
    CHROMA_PORT = 8000

VERIFY_SSL = VERIFY_SSL_STR.lower() == 'true'
if not VERIFY_SSL and EMBEDDING_API_URL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_embedding_model_cache = None
def get_embedding_model():
    global _embedding_model_cache
    if _embedding_model_cache is None:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading local sentence transformer model: {MODEL_NAME_FOR_EMBEDDING}")
            _embedding_model_cache = SentenceTransformer(MODEL_NAME_FOR_EMBEDDING)
            print("Local embedding model loaded.")
        except ImportError:
            print("Warning: sentence_transformers library not found. Local embedding will not work.")
            _embedding_model_cache = "error"
        except Exception as e:
            print(f"Error loading local SentenceTransformer model: {e}")
            _embedding_model_cache = "error"
    return _embedding_model_cache if _embedding_model_cache != "error" else None

def get_remote_embedding_batch(texts_batch):
    if not EMBEDDING_API_URL:
        print("Error: EMBEDDING_API_URL not configured for remote embedding.")
        return [None] * len(texts_batch) 

    if not texts_batch: return []
    
    try:
        print(f"Calling remote embedding API for {len(texts_batch)} texts...")
        response = requests.post(EMBEDDING_API_URL, json={"texts": texts_batch}, verify=VERIFY_SSL, timeout=60)
        response.raise_for_status()
        embeddings_data = response.json().get("embeddings", [])
        if len(embeddings_data) == len(texts_batch):
            return [np.array(emb) if emb is not None else None for emb in embeddings_data]
        else:
            print(f"Warning: Mismatch in remote embeddings count. Expected {len(texts_batch)}, got {len(embeddings_data)}.")
            return [None] * len(texts_batch)
    except Exception as e:
        print(f"Error calling remote embedding API: {e}")
        traceback.print_exc() 
        return [None] * len(texts_batch)

def generate_embedding_for_skills(skills_list):
    if not skills_list or not isinstance(skills_list, list):
        return None
    
    valid_skills = [s for s in skills_list if isinstance(s, str) and s.strip()]
    if not valid_skills:
        return None

    skill_embeddings_np = []

    if EMBEDDING_API_URL:
        print(f"Attempting remote embedding for {len(valid_skills)} skills.")
        batch_size = 32 
        for i in range(0, len(valid_skills), batch_size):
            batch = valid_skills[i:i+batch_size]
            remote_embs = get_remote_embedding_batch(batch)
            skill_embeddings_np.extend([emb for emb in remote_embs if emb is not None])
    else:
        print("EMBEDDING_API_URL not set. Attempting local embedding.")
        model = get_embedding_model()
        if model:
            try:
                raw_embeddings = model.encode(valid_skills, show_progress_bar=False)
                skill_embeddings_np = [np.array(emb) for emb in raw_embeddings]
            except Exception as e:
                print(f"Error during local batch skill embedding: {e}")
        else:
            print("Local embedding model not available. Cannot generate skill embeddings.")
            return None

    if not skill_embeddings_np:
        print("No valid skill embeddings were generated.")
        return None

    average_embedding = np.mean(skill_embeddings_np, axis=0)
    norm = np.linalg.norm(average_embedding)
    
    return (average_embedding / norm if norm > 0 else average_embedding).tolist()

def cosine_similarity_np(vec1, vec2):
    if vec1 is None or vec2 is None: return 0.0
    vec1_np, vec2_np = np.array(vec1), np.array(vec2)
    if vec1_np.shape != vec2_np.shape or np.all(vec1_np==0) or np.all(vec2_np==0):
        return 0.0
    dot_product = np.dot(vec1_np, vec2_np)
    norm_vec1, norm_vec2 = np.linalg.norm(vec1_np), np.linalg.norm(vec2_np)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

def explain_job_match(cv_skills, job_skills_from_meta, cv_embedding_overall, job_embedding_from_chroma):
    if not cv_skills or job_embedding_from_chroma is None:
        return []
        
    valid_cv_skills = [s for s in cv_skills if isinstance(s, str) and s.strip()]
    if not valid_cv_skills:
        return []
        
    cv_individual_skill_vectors_list = []
    if EMBEDDING_API_URL:
        cv_individual_skill_vectors_list = get_remote_embedding_batch(valid_cv_skills)
    else:
        model = get_embedding_model()
        if model:
            try:
                raw_embeddings = model.encode(valid_cv_skills, show_progress_bar=False)
                cv_individual_skill_vectors_list = [np.array(e) for e in raw_embeddings]
            except Exception as e:
                print(f"Error embedding individual CV skills locally for explanation: {e}")
        else:
            print("Local model not available for CV skill embedding in explain_job_match.")
            return []

    successfully_embedded_cv_skills = []
    successfully_embedded_cv_vectors = []
    for i, skill_text in enumerate(valid_cv_skills):
        if i < len(cv_individual_skill_vectors_list) and cv_individual_skill_vectors_list[i] is not None:
            successfully_embedded_cv_skills.append(skill_text)
            successfully_embedded_cv_vectors.append(cv_individual_skill_vectors_list[i])

    if not successfully_embedded_cv_vectors:
        print("No CV skills could be embedded for explanation.")
        return []

    skill_contributions = []
    job_emb_np = np.array(job_embedding_from_chroma)

    for i, skill_text in enumerate(successfully_embedded_cv_skills):
        skill_vector = successfully_embedded_cv_vectors[i]
        similarity = cosine_similarity_np(skill_vector, job_emb_np) 
        skill_contributions.append((skill_text, similarity))

    skill_contributions.sort(key=lambda x: x[1], reverse=True)
    return skill_contributions[:EXPLAIN_TOP_N_CONTRIBUTING_SKILLS]

def find_similar_jobs(cv_skill_embedding, cv_skills, top_n=None, active_only=True):
    if top_n is None:
        top_n = TOP_N_RESULTS_DEFAULT

    if cv_skill_embedding is None:
        return [], "CV Skill Embedding is missing."

    if not all([CHROMA_HOST, CHROMA_PORT_STR, COLLECTION_NAME]):
        return [], "ChromaDB connection details (host, port, collection) are not fully configured."

    try:
        chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=ChromaSettings(anonymized_telemetry=False))
        
        try:
            collection = chroma_client.get_collection(COLLECTION_NAME)
            print(f"Connected to ChromaDB collection: {COLLECTION_NAME}")
        except Exception as conn_err:
            return [], f"Failed to get ChromaDB collection '{COLLECTION_NAME}': {conn_err}"

        where_clause = {"Status": "active"} if active_only else None
        if active_only: print("Filtering for active jobs only in ChromaDB.")
        
        query_embedding_list = cv_skill_embedding if isinstance(cv_skill_embedding, list) else cv_skill_embedding.tolist()

        print(f"Querying ChromaDB with top_n={top_n}...")
        results = collection.query(
            query_embeddings=[query_embedding_list],
            n_results=top_n,
            include=["metadatas", "distances", "documents", "embeddings"], 
            where=where_clause
        )
        
        matches = []
        if results and results.get('ids') and results['ids'] and isinstance(results['ids'][0], list):
            num_results = len(results['ids'][0])
            print(f"ChromaDB returned {num_results} raw results.")

            distances_list = results.get('distances', [[]])[0] if results.get('distances') and results['distances'] else [None] * num_results
            metadatas_list = results.get('metadatas', [[]])[0] if results.get('metadatas') and results['metadatas'] else [{}] * num_results
            documents_list = results.get('documents', [[]])[0] if results.get('documents') and results['documents'] else [""] * num_results
            embeddings_list = results.get('embeddings', [[]])[0] if results.get('embeddings') and results['embeddings'] else [None] * num_results

            if not (len(distances_list) == num_results and \
                    len(metadatas_list) == num_results and \
                    len(documents_list) == num_results and \
                    len(embeddings_list) == num_results):
                print("Warning: Mismatch in lengths of returned lists from ChromaDB query. Results might be incomplete.")
                min_len = min(len(distances_list), len(metadatas_list), len(documents_list), len(embeddings_list), num_results)
                if min_len < num_results:
                    print(f"Adjusting iteration from {num_results} to {min_len} due to list length mismatch.")
                num_results = min_len

            for i in range(num_results):
                chroma_id = results['ids'][0][i]
                distance = distances_list[i]
                metadata = metadatas_list[i] if isinstance(metadatas_list[i], dict) else {}
                document_text = documents_list[i] if isinstance(documents_list[i], str) else ""
                job_embedding_item = embeddings_list[i]
                
                job_embedding = None
                if job_embedding_item is not None and (isinstance(job_embedding_item, list) or isinstance(job_embedding_item, np.ndarray)):
                    job_embedding = np.array(job_embedding_item)

                if distance is None:
                    similarity_score_percent = 0.0 
                else:
                    clamped_distance = min(max(float(distance), 0.0), 1.0) 
                    similarity_score_percent = (1.0 - clamped_distance) * 100.0
                
                job_skills_str = metadata.get('Skills_Json_Str', '[]')
                try:
                    job_skills_list = json.loads(job_skills_str) if isinstance(job_skills_str, str) else []
                except json.JSONDecodeError:
                    job_skills_list = []
                    print(f"Warning: Could not parse Skills_Json_Str for job {chroma_id}: {job_skills_str[:50]}")

                contributing_cv_skills = []
                if job_embedding is not None and cv_skills: 
                     contributing_cv_skills = explain_job_match(cv_skills, job_skills_list, cv_skill_embedding, job_embedding)

                # --- THIS IS THE KEY CHANGE ---
                match_data = {**metadata} # Spread all items from metadata
                match_data.update({        # Add or override specific calculated/fixed fields
                    "chroma_id": chroma_id,
                    "score": similarity_score_percent,
                    "document_text": document_text, 
                    "job_skills": job_skills_list, 
                    "contributing_skills": contributing_cv_skills,
                    "url": metadata.get('Application_URL', metadata.get('URL', '#')) 
                })
                # --- END OF KEY CHANGE ---
                matches.append(match_data)
            
            print(f"Processed {len(matches)} matches from ChromaDB results.")
        else:
            print("No results or unexpected format from ChromaDB query. 'results.ids[0]' might be empty or not a list.")

        return matches, "ChromaDB Vector Search"

    except Exception as e:
        print(f"Error during ChromaDB search: {e}")
        traceback.print_exc()
        return [], f"ChromaDB Search Failed: {e}"

if __name__ == "__main__":
    print("cv_match.py loaded for direct execution test.")
    if not all([EMBEDDING_API_URL, CHROMA_HOST, CHROMA_PORT_STR, COLLECTION_NAME]):
        print("Skipping test: Missing one or more required .env variables for full test.")
    else:
        sample_cv_skills_list = ["Python", "machine learning", "data analysis", "communication", "projektledelse", "SQL"]
        print(f"Test CV Skills: {sample_cv_skills_list}")
        
        test_cv_skill_embedding = generate_embedding_for_skills(sample_cv_skills_list)
        
        if test_cv_skill_embedding:
            print(f"Generated test CV embedding (first 5 dims): {test_cv_skill_embedding[:5]}")
            job_matches_found, message_status = find_similar_jobs(test_cv_skill_embedding, sample_cv_skills_list, top_n=5)
            print(f"Search Status: {message_status}")
            if job_matches_found:
                print(f"Found {len(job_matches_found)} job matches:")
                for i_match, match_item in enumerate(job_matches_found):
                    print(f"\n  {i_match+1}. Title: {match_item.get('Title', 'N/A')}") # Use .get() for safety
                    print(f"     Score: {match_item.get('score', 0.0):.2f}%")
                    print(f"     Company: {match_item.get('Company', 'N/A')}, Area: {match_item.get('Area', 'N/A')}")
                    print(f"     Category: {match_item.get('Category', 'N/A')}") # Test: print Category
                    print(f"     Contributing CV Skills: {[(s_item[0], round(s_item[1], 3)) for s_item in match_item.get('contributing_skills', [])]}")
            else:
                print("No job matches found for the test CV skills.")
        else:
            print("Failed to generate embedding for test CV skills.")
