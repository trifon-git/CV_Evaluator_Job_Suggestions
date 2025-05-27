import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pymongo import MongoClient
from chromadb import HttpClient, Settings as ChromaSettings
import time
import traceback 

import requests 
import urllib3  

load_dotenv()

# --- Helper Function for Logging ---
def log_message_step4(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - [STEP4] - {message}")

# --- Configuration ---
MODEL_NAME_FOR_EMBEDDING_STEP4 = os.getenv('MODEL_NAME', 'paraphrase-multilingual-mpnet-base-v2') # Local model name remains specific
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME_MONGO_STEP4 = os.getenv('MONGO_DB_NAME') # Mongo DB/Collection names can remain specific if needed
COLLECTION_NAME_MONGO_STEP4 = os.getenv('MONGO_COLLECTION')

CHROMA_HOST = os.getenv('CHROMA_HOST')
CHROMA_PORT_STR_STEP4 = os.getenv('CHROMA_PORT') # Chroma port can remain specific
CHROMA_COLLECTION_NAME_STEP4 = os.getenv('CHROMA_COLLECTION') # Chroma collection name can remain specific
BATCH_SIZE_CHROMA = int(os.getenv('BATCH_SIZE_CHROMA_IMPORT', '100'))

USE_REMOTE_EMBEDDING = os.getenv('USE_REMOTE_EMBEDDING', 'false').lower() == 'true'
EMBEDDING_API_URL = os.getenv('EMBEDDING_API_URL') # e.g., http://localhost:8001/embed
VERIFY_SSL = os.getenv('VERIFY_SSL', 'true').lower() == 'true'

if not VERIFY_SSL and EMBEDDING_API_URL: # Only disable warnings if remote API is used and SSL verify is off
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    log_message_step4("SSL verification is DISABLED for remote API calls.")


try:
    CHROMA_PORT = int(CHROMA_PORT_STR_STEP4) if CHROMA_PORT_STR_STEP4 else 8000
except ValueError: CHROMA_PORT = 8000

_local_embedding_model_step4_cache = None

def get_local_embedding_model_for_step4():
    global _local_embedding_model_step4_cache
    if _local_embedding_model_step4_cache is None:
        log_message_step4(f"Loading local model: {MODEL_NAME_FOR_EMBEDDING_STEP4}")
        try:
            _local_embedding_model_step4_cache = SentenceTransformer(MODEL_NAME_FOR_EMBEDDING_STEP4)
            log_message_step4("Local embedding model loaded for [STEP4].")
        except Exception as e:
            log_message_step4(f"ERROR loading local model: {e}"); _local_embedding_model_step4_cache = "ERROR"
    return _local_embedding_model_step4_cache if _local_embedding_model_step4_cache != "ERROR" else None

def generate_average_skill_embedding_local_step4(skills_list):
    model = get_local_embedding_model_for_step4()
    if model is None: return None
    if not skills_list or not isinstance(skills_list, list): return None
    valid_skills = [s for s in skills_list if isinstance(s, str) and s.strip()]
    if not valid_skills: return None
    try:
        embeddings_np = model.encode(valid_skills, show_progress_bar=False)
        if not hasattr(embeddings_np, 'size') or not embeddings_np.size: return None
        avg_emb_np = np.mean(embeddings_np, axis=0); norm = np.linalg.norm(avg_emb_np)
        return (avg_emb_np / norm if norm > 0 else avg_emb_np).tolist()
    except Exception as e: log_message_step4(f"ERROR generating local embeddings: {str(e)[:100]}..."); return None

def get_remote_embeddings_step4(batch_skill_lists):
    """
    Calls a remote API to get embeddings for a batch of skill lists.
    Expects each skill list to be joined into a single string for the API.
    Uses general EMBEDDING_API_URL and VERIFY_SSL from .env.
    """
    if not EMBEDDING_API_URL: # Uses general env var
        log_message_step4("Remote embedding API URL not configured. Cannot use remote embedding.")
        return None

    texts_for_api = [", ".join(skills) for skills in batch_skill_lists]
    if not texts_for_api:
        return []

    payload = {"texts": texts_for_api}
    log_message_step4(f"Calling remote embedding API at {EMBEDDING_API_URL} for {len(texts_for_api)} items.")

    try:
        response = requests.post(EMBEDDING_API_URL, json=payload, verify=VERIFY_SSL, timeout=60) # Uses general env vars
        response.raise_for_status()
        response_data = response.json()
        embeddings = response_data.get("embeddings")

        if not embeddings:
            log_message_step4("Warning: Remote API returned no embeddings in the 'embeddings' key.")
            return None
        if len(embeddings) != len(texts_for_api):
            log_message_step4(f"Warning: Mismatch in remote embeddings count. Expected {len(texts_for_api)}, got {len(embeddings)}.")
            return None

        log_message_step4(f"Successfully received {len(embeddings)} embeddings from remote API.")
        return embeddings
    except requests.exceptions.RequestException as e:
        log_message_step4(f"ERROR calling remote embedding API: {e}")
        return None
    except json.JSONDecodeError as e:
        log_message_step4(f"ERROR decoding JSON response from remote API: {e}. Response text: {response.text[:200]}")
        return None
    except Exception as e:
        log_message_step4(f"Unexpected ERROR during remote embedding call: {e}")
        return None


def generate_embeddings_for_batch(batch_skill_lists):
    if not batch_skill_lists: return []

    if USE_REMOTE_EMBEDDING and EMBEDDING_API_URL: # Uses general env vars
        log_message_step4("Attempting to use remote embedding service...")
        remote_embeddings = get_remote_embeddings_step4(batch_skill_lists)
        if remote_embeddings is not None:
            return remote_embeddings
        else:
            log_message_step4("Remote embedding failed or not configured properly. Falling back to local model.")

    log_message_step4("Using local model for embeddings.")
    embeddings_for_batch = []
    for skills in batch_skill_lists:
        embedding = generate_average_skill_embedding_local_step4(skills)
        embeddings_for_batch.append(embedding)
    return embeddings_for_batch


def main_import_to_chroma():
    log_message_step4("Starting MongoDB to ChromaDB import process...")
    if USE_REMOTE_EMBEDDING: # Uses general env var
        log_message_step4(f"Remote embedding is ENABLED. API URL: {EMBEDDING_API_URL}")
    else:
        log_message_step4("Remote embedding is DISABLED. Using local model.")

    mongo_client_obj, chroma_client_obj = None, None
    try:
        mongo_client_obj = MongoClient(MONGO_URI); mongo_db = mongo_client_obj[DB_NAME_MONGO_STEP4]
        mongo_jobs_collection = mongo_db[COLLECTION_NAME_MONGO_STEP4]; log_message_step4("Connected to MongoDB.")

        chroma_client_obj = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=ChromaSettings(anonymized_telemetry=False))
        try:
            chroma_collection = chroma_client_obj.get_collection(name=CHROMA_COLLECTION_NAME_STEP4)
            if chroma_collection is None:
                raise ValueError("Chroma collection could not be retrieved.")
            collection_metadata = chroma_collection.metadata or {}
            if collection_metadata.get("hnsw:space") != "cosine":
                log_message_step4(f"WARNING: Chroma Collection '{CHROMA_COLLECTION_NAME_STEP4}' exists but metric is '{collection_metadata.get('hnsw:space', 'Unknown')}'. 'cosine' is recommended.")
        except Exception as e_get_coll:
            log_message_step4(f"Chroma Collection '{CHROMA_COLLECTION_NAME_STEP4}' not found or error accessing ({e_get_coll}). Creating with 'cosine' metric.")
            chroma_collection = chroma_client_obj.create_collection(name=CHROMA_COLLECTION_NAME_STEP4, metadata={"hnsw:space": "cosine"})
            if chroma_collection is None:
                log_message_step4("ERROR: Failed to create Chroma collection.")
                return
            log_message_step4(f"Using ChromaDB collection: {CHROMA_COLLECTION_NAME_STEP4} (Metric: {chroma_collection.metadata.get('hnsw:space', 'Unknown')})")


        try: existing_chroma_ids = set(chroma_collection.get(include=[])['ids'])
        except Exception: existing_chroma_ids = set()
        log_message_step4(f"Found {len(existing_chroma_ids)} existing document IDs in ChromaDB.")

        mongo_query = {
            "Status": "active",
            "$or": [
                {"Skills": {"$exists": True, "$ne": [], "$ne": None}},
                {"skills": {"$exists": True, "$ne": [], "$ne": None}}
            ],
        }
        log_message_step4(f"MongoDB query for jobs: {json.dumps(mongo_query)}")
        jobs_to_process_cursor = mongo_jobs_collection.find(mongo_query)

        jobs_for_chroma = []
        for job in jobs_to_process_cursor:
            if str(job["_id"]) not in existing_chroma_ids:
                jobs_for_chroma.append(job)

        if not jobs_for_chroma: log_message_step4("No new jobs found in MongoDB matching criteria to import. Exiting."); return
        log_message_step4(f"Found {len(jobs_for_chroma)} new jobs to process for ChromaDB.")

        total_imported_count = 0
        for i in tqdm(range(0, len(jobs_for_chroma), BATCH_SIZE_CHROMA), desc="Importing to ChromaDB"):
            batch_jobs_mongo = jobs_for_chroma[i:i+BATCH_SIZE_CHROMA]
            batch_ids_to_add, batch_skill_lists_for_embedding, batch_metadatas_to_add = [], [], []

            for job_doc in batch_jobs_mongo:
                skills_list_raw = job_doc.get("Skills")
                if not skills_list_raw or not isinstance(skills_list_raw, list) or not any(s.strip() for s in skills_list_raw if isinstance(s, str)):
                    skills_list_raw = job_doc.get("skills", [])

                skills_list_cleaned = [s.strip() for s in skills_list_raw if isinstance(s, str) and s.strip()]

                if not skills_list_cleaned:
                    log_message_step4(f"Job ID {job_doc['_id']} has no valid skills in 'Skills' or 'skills' field after cleaning. Skipping.")
                    continue

                batch_ids_to_add.append(str(job_doc["_id"]))
                batch_skill_lists_for_embedding.append(skills_list_cleaned)

                # --- METADATA PREPARATION START ---
                metadata = {
                    "_id": str(job_doc.get("_id")),
                    "job_id": job_doc.get("job_id", "N/A"),
                    "Title": job_doc.get("Title", "N/A"),
                    "Company": job_doc.get("Company", "N/A"),
                    "URL": job_doc.get("URL", "#"),
                    "Application_URL": job_doc.get("Application_URL", "#"),
                    "Area": job_doc.get("Area", "N/A"),
                    "Category": job_doc.get("Category", "N/A"),
                    "Status": job_doc.get("Status", "unknown"),
                    "Experience_Level_Required": job_doc.get("Experience_Level_Required", job_doc.get("experience_level_required", "Not specified")),
                    "Education_Level_Preferred": job_doc.get("Education_Level_Preferred", job_doc.get("education_level_preferred", "Not specified")),
                    "Job_Type": job_doc.get("Job_Type", job_doc.get("job_type", "Not specified")),
                    "Detected_Ad_Language": job_doc.get("Detected_Ad_Language", job_doc.get("detected_ad_language", "Unknown"))
                }
                metadata["Skills_Json_Str"] = json.dumps(skills_list_cleaned) if skills_list_cleaned else "[]"
                lang_req_list_raw = job_doc.get("Language_Requirements", job_doc.get("language_requirements", []))
                processed_languages_for_json = []
                if isinstance(lang_req_list_raw, list):
                    for lang_item in lang_req_list_raw:
                        if isinstance(lang_item, dict):
                            lang = lang_item.get("language")
                            prof = lang_item.get("proficiency")
                            lang_clean_for_json = None
                            prof_clean_for_json = None
                            if lang and isinstance(lang, str) and lang.strip():
                                lang_clean = lang.strip().lower()
                                lang_clean_for_json = lang.strip()
                                if prof and isinstance(prof, str) and prof.strip():
                                    prof_clean = prof.strip().lower()
                                    prof_clean_for_json = prof.strip()
                                    metadata[f"lang_{lang_clean}_proficiency"] = prof_clean
                                else:
                                     metadata[f"lang_{lang_clean}_proficiency"] = "unspecified"
                            if lang_clean_for_json:
                                processed_languages_for_json.append({
                                    "language": lang_clean_for_json,
                                    "proficiency": prof_clean_for_json if prof_clean_for_json else "Unspecified"
                                })
                metadata["Language_Requirements_Json_Str"] = json.dumps(processed_languages_for_json) if processed_languages_for_json else "[]"
                # --- METADATA PREPARATION END ---
                batch_metadatas_to_add.append(metadata)

            if not batch_ids_to_add: continue

            batch_embeddings_list_type = generate_embeddings_for_batch(batch_skill_lists_for_embedding)

            final_ids, final_embeddings, final_metadatas = [], [], []
            for idx, emb in enumerate(batch_embeddings_list_type):
                if emb is not None and isinstance(emb, list):
                    final_ids.append(batch_ids_to_add[idx])
                    final_embeddings.append(emb)
                    final_metadatas.append(batch_metadatas_to_add[idx])
                else:
                    log_message_step4(f"Skipping job ID {batch_ids_to_add[idx]} due to embedding failure (embedding was None or not a list).")

            if final_ids:
                try:
                    chroma_collection.add(ids=final_ids, embeddings=final_embeddings, metadatas=final_metadatas)
                    total_imported_count += len(final_ids)
                    log_message_step4(f"Successfully added batch of {len(final_ids)} jobs to ChromaDB.")
                except Exception as e_add:
                    log_message_step4(f"ERROR adding batch to ChromaDB: {e_add}")
                    log_message_step4(f"Batch details: IDs ({len(final_ids)}), Embeddings ({len(final_embeddings)}), Metadatas ({len(final_metadatas)})")
                    if final_ids:
                        log_message_step4(f"First ID: {final_ids[0]}, First metadata: {str(final_metadatas[0])[:200]}")


        log_message_step4(f"Import finished. Total new jobs added to ChromaDB: {total_imported_count}")

    except Exception as e:
        log_message_step4(f"CRITICAL ERROR during ChromaDB import: {e}")
        traceback.print_exc()
    finally:
        if mongo_client_obj: mongo_client_obj.close()

if __name__ == "__main__":
    main_import_to_chroma()
