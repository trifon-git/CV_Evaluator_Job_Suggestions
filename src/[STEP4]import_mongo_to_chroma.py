import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pymongo import MongoClient
from chromadb import HttpClient, Settings as ChromaSettings
import chromadb # Import for chromadb.errors
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
MODEL_NAME_FOR_EMBEDDING_STEP4 = os.getenv('MODEL_NAME', 'paraphrase-multilingual-mpnet-base-v2')
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME_MONGO_STEP4 = os.getenv('MONGO_DB_NAME')
COLLECTION_NAME_MONGO_STEP4 = os.getenv('MONGO_COLLECTION')

CHROMA_HOST = os.getenv('CHROMA_HOST')
CHROMA_PORT_STR_STEP4 = os.getenv('CHROMA_PORT')
CHROMA_COLLECTION_NAME_STEP4 = os.getenv('CHROMA_COLLECTION')
BATCH_SIZE_CHROMA = int(os.getenv('BATCH_SIZE_CHROMA_IMPORT', '100'))

USE_REMOTE_EMBEDDING = os.getenv('USE_REMOTE_EMBEDDING', 'false').lower() == 'true'
EMBEDDING_API_URL = os.getenv('EMBEDDING_API_URL')
VERIFY_SSL = os.getenv('VERIFY_SSL', 'true').lower() == 'true'

if not VERIFY_SSL and EMBEDDING_API_URL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    log_message_step4("SSL verification is DISABLED for remote API calls.")

try:
    CHROMA_PORT = int(CHROMA_PORT_STR_STEP4) if CHROMA_PORT_STR_STEP4 else 8000
except ValueError: 
    CHROMA_PORT = 8000
    log_message_step4(f"Warning: Invalid CHROMA_PORT '{CHROMA_PORT_STR_STEP4}', defaulting to {CHROMA_PORT}.")

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
    if not EMBEDDING_API_URL:
        log_message_step4("Remote embedding API URL not configured. Cannot use remote embedding.")
        return None
    texts_for_api = [", ".join(skills) for skills in batch_skill_lists if skills] 
    if not texts_for_api: return []
    payload = {"texts": texts_for_api}
    log_message_step4(f"Calling remote embedding API at {EMBEDDING_API_URL} for {len(texts_for_api)} items.")
    try:
        response = requests.post(EMBEDDING_API_URL, json=payload, verify=VERIFY_SSL, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        embeddings = response_data.get("embeddings")
        if not embeddings or len(embeddings) != len(texts_for_api):
            log_message_step4(f"Warning: Remote API embedding mismatch or empty. Expected {len(texts_for_api)}, got {len(embeddings if embeddings else [])}.")
            return None
        log_message_step4(f"Successfully received {len(embeddings)} embeddings from remote API.")
        return embeddings
    except requests.exceptions.RequestException as e: log_message_step4(f"ERROR calling remote embedding API: {e}"); return None
    except json.JSONDecodeError as e: log_message_step4(f"ERROR decoding JSON from remote API: {e}. Response: {response.text[:200]}"); return None
    except Exception as e: log_message_step4(f"Unexpected ERROR during remote embedding: {e}"); return None

def generate_embeddings_for_batch(batch_skill_lists):
    if not batch_skill_lists: return []
    if USE_REMOTE_EMBEDDING and EMBEDDING_API_URL:
        log_message_step4("Attempting to use remote embedding service...")
        remote_embeddings = get_remote_embeddings_step4(batch_skill_lists)
        if remote_embeddings is not None: return remote_embeddings
        else: log_message_step4("Remote embedding failed/not configured. Falling back to local.")
    log_message_step4("Using local model for embeddings.")
    return [generate_average_skill_embedding_local_step4(skills) for skills in batch_skill_lists]

def main_import_to_chroma():
    log_message_step4("Starting MongoDB to ChromaDB import process...")
    if USE_REMOTE_EMBEDDING: log_message_step4(f"Remote embedding is ENABLED. API URL: {EMBEDDING_API_URL}")
    else: log_message_step4("Remote embedding is DISABLED. Using local model.")

    mongo_client_obj, chroma_client_obj = None, None
    try:
        mongo_client_obj = MongoClient(MONGO_URI); mongo_db = mongo_client_obj[DB_NAME_MONGO_STEP4]
        mongo_jobs_collection = mongo_db[COLLECTION_NAME_MONGO_STEP4]; log_message_step4("Connected to MongoDB.")

        chroma_client_obj = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=ChromaSettings(anonymized_telemetry=False))
        
        chroma_collection = None
        desired_metric = "cosine" # This is the metric we want the collection to use.
        effective_metric = "" # To store the actual metric used

        try:
            log_message_step4(f"Attempting to get Chroma Collection: '{CHROMA_COLLECTION_NAME_STEP4}'")
            # Try to get the collection first to check its existing metric
            temp_collection = chroma_client_obj.get_collection(name=CHROMA_COLLECTION_NAME_STEP4)
            
            # Collection exists, check its metric
            collection_metadata = temp_collection.metadata 
            current_metric = "l2" # Chroma's default if not specified in metadata
            if collection_metadata and "hnsw:space" in collection_metadata:
                current_metric = collection_metadata["hnsw:space"]

            if current_metric == desired_metric:
                log_message_step4(f"Chroma Collection '{CHROMA_COLLECTION_NAME_STEP4}' already exists with the desired '{desired_metric}' metric.")
                chroma_collection = temp_collection
                effective_metric = desired_metric
            else:
                log_message_step4(f"WARNING: Chroma Collection '{CHROMA_COLLECTION_NAME_STEP4}' exists with metric '{current_metric}', but '{desired_metric}' is required.")
                log_message_step4(f"Deleting existing collection '{CHROMA_COLLECTION_NAME_STEP4}' to recreate with '{desired_metric}' metric.")
                try:
                    chroma_client_obj.delete_collection(name=CHROMA_COLLECTION_NAME_STEP4)
                    log_message_step4(f"Successfully deleted collection '{CHROMA_COLLECTION_NAME_STEP4}'.")
                    chroma_collection = None # Signal to recreate it
                except Exception as e_delete:
                    log_message_step4(f"ERROR: Failed to delete existing collection '{CHROMA_COLLECTION_NAME_STEP4}': {e_delete}. Exiting.")
                    return
        
        except chromadb.errors.CollectionNotDefinedError: # Specific ChromaDB error for not found
             log_message_step4(f"Chroma Collection '{CHROMA_COLLECTION_NAME_STEP4}' not found. Will be created with '{desired_metric}' metric.")
             chroma_collection = None # Signal to create it
        except Exception as e_get_coll: # Catch other potential errors during get_collection
            log_message_step4(f"An unexpected error occurred while trying to get collection '{CHROMA_COLLECTION_NAME_STEP4}': {e_get_coll}. Assuming it needs to be created.")
            chroma_collection = None # Signal to create it

        # If collection is None (either not found, or was deleted due to metric mismatch), create it
        if chroma_collection is None:
            try:
                log_message_step4(f"Creating Chroma Collection '{CHROMA_COLLECTION_NAME_STEP4}' with '{desired_metric}' metric.")
                chroma_collection = chroma_client_obj.create_collection(
                    name=CHROMA_COLLECTION_NAME_STEP4, 
                    metadata={"hnsw:space": desired_metric}
                )
                effective_metric = desired_metric 
                log_message_step4(f"Successfully created Chroma Collection '{CHROMA_COLLECTION_NAME_STEP4}' with '{desired_metric}' metric.")
            except Exception as e_create_coll:
                log_message_step4(f"ERROR: Critical failure creating Chroma Collection '{CHROMA_COLLECTION_NAME_STEP4}': {e_create_coll}. Exiting.")
                return
        
        if not chroma_collection: # Should not happen if creation was successful or get was successful
            log_message_step4(f"ERROR: Chroma collection object is unexpectedly None after get/create attempts. Exiting.")
            return

        log_message_step4(f"Using ChromaDB collection: '{CHROMA_COLLECTION_NAME_STEP4}' (Effective Metric: '{effective_metric}')")


        try: existing_chroma_ids = set(chroma_collection.get(include=[])['ids'])
        except Exception: existing_chroma_ids = set() # Should be empty if we just created/recreated it
        log_message_step4(f"Found {len(existing_chroma_ids)} existing document IDs in ChromaDB.")

        mongo_query = {
            "Status": "active",
            "html_content": {"$exists": True, "$ne": None, "$ne": ""}, 
            "$or": [
                {"Skills": {"$exists": True, "$ne": [], "$ne": None}},
                {"skills": {"$exists": True, "$ne": [], "$ne": None}}
            ],
        }
        jobs_to_process_cursor = mongo_jobs_collection.find(mongo_query, {"html_content": 1, "_id": 1, "Skills":1, "skills":1, "job_id":1, "Title":1, "Company":1, "URL":1, "Application_URL":1, "Area":1, "Category":1, "Status":1, "Experience_Level_Required":1, "experience_level_required":1, "Education_Level_Preferred":1, "education_level_preferred":1, "Job_Type":1, "job_type":1, "Detected_Ad_Language":1, "detected_ad_language":1, "Language_Requirements":1, "language_requirements":1 })

        jobs_for_chroma = [job for job in jobs_to_process_cursor if str(job["_id"]) not in existing_chroma_ids]
        if not jobs_for_chroma: log_message_step4("No new jobs found in MongoDB matching criteria to import. Exiting."); return
        log_message_step4(f"Found {len(jobs_for_chroma)} new jobs to process for ChromaDB.")

        total_imported_count = 0
        for i in tqdm(range(0, len(jobs_for_chroma), BATCH_SIZE_CHROMA), desc="Importing to ChromaDB"):
            batch_jobs_mongo = jobs_for_chroma[i:i+BATCH_SIZE_CHROMA]
            batch_ids_to_add, batch_skill_lists_for_embedding, batch_metadatas_to_add, batch_documents_to_add = [], [], [], []

            for job_doc in batch_jobs_mongo:
                skills_list_raw = job_doc.get("Skills", job_doc.get("skills", []))
                skills_list_cleaned = [s.strip() for s in skills_list_raw if isinstance(s, str) and s.strip()]

                if not skills_list_cleaned:
                    log_message_step4(f"Job ID {job_doc['_id']} has no valid skills. Skipping.")
                    continue
                
                document_text = job_doc.get("html_content", "") 
                if not document_text:
                    log_message_step4(f"Warning: Job ID {job_doc['_id']} has empty html_content. Using placeholder document.")
                    document_text = "No job description text available."

                batch_ids_to_add.append(str(job_doc["_id"]))
                batch_skill_lists_for_embedding.append(skills_list_cleaned)
                batch_documents_to_add.append(document_text)

                metadata = {
                    "_id": str(job_doc.get("_id")), "job_id": job_doc.get("job_id", "N/A"),
                    "Title": job_doc.get("Title", "N/A"), "Company": job_doc.get("Company", "N/A"),
                    "URL": job_doc.get("URL", "#"), "Application_URL": job_doc.get("Application_URL", "#"),
                    "Area": job_doc.get("Area", "N/A"), "Category": job_doc.get("Category", "N/A"),
                    "Status": job_doc.get("Status", "unknown"),
                    "Experience_Level_Required": job_doc.get("Experience_Level_Required", job_doc.get("experience_level_required", "Not specified")),
                    "Education_Level_Preferred": job_doc.get("Education_Level_Preferred", job_doc.get("education_level_preferred", "Not specified")),
                    "Job_Type": job_doc.get("Job_Type", job_doc.get("job_type", "Not specified")),
                    "Detected_Ad_Language": job_doc.get("Detected_Ad_Language", job_doc.get("detected_ad_language", "Unknown"))
                }
                metadata["Skills_Json_Str"] = json.dumps(skills_list_cleaned)
                lang_req_list_raw = job_doc.get("Language_Requirements", job_doc.get("language_requirements", []))
                processed_languages_for_json = []
                if isinstance(lang_req_list_raw, list):
                    for lang_item in lang_req_list_raw:
                        if isinstance(lang_item, dict) and lang_item.get("language"):
                            lang_clean = lang_item["language"].strip().lower()
                            prof_clean = (lang_item.get("proficiency") or "unspecified").strip().lower()
                            metadata[f"lang_{lang_clean}_proficiency"] = prof_clean
                            processed_languages_for_json.append({"language": lang_item["language"].strip(), "proficiency": (lang_item.get("proficiency") or "Unspecified").strip()})
                metadata["Language_Requirements_Json_Str"] = json.dumps(processed_languages_for_json)
                batch_metadatas_to_add.append(metadata)

            if not batch_ids_to_add: continue

            batch_embeddings_list_type = generate_embeddings_for_batch(batch_skill_lists_for_embedding)

            final_ids, final_embeddings, final_metadatas, final_documents = [], [], [], []
            for idx, emb in enumerate(batch_embeddings_list_type):
                if emb is not None and isinstance(emb, list):
                    final_ids.append(batch_ids_to_add[idx])
                    final_embeddings.append(emb)
                    final_metadatas.append(batch_metadatas_to_add[idx])
                    final_documents.append(batch_documents_to_add[idx])
                else:
                    log_message_step4(f"Skipping job ID {batch_ids_to_add[idx]} due to embedding failure.")

            if final_ids:
                try:
                    chroma_collection.add(
                        ids=final_ids, 
                        embeddings=final_embeddings, 
                        metadatas=final_metadatas,
                        documents=final_documents 
                    )
                    total_imported_count += len(final_ids)
                    log_message_step4(f"Successfully added batch of {len(final_ids)} jobs (with documents) to ChromaDB.")
                except Exception as e_add:
                    log_message_step4(f"ERROR adding batch to ChromaDB: {e_add}")
                    log_message_step4(f"Batch details: IDs ({len(final_ids)}), Embeddings ({len(final_embeddings)}), Metadatas ({len(final_metadatas)}), Documents ({len(final_documents)})")
                    if final_ids: log_message_step4(f"First ID: {final_ids[0]}, First metadata: {str(final_metadatas[0])[:200]}, First document (preview): {str(final_documents[0])[:100]}...")
        log_message_step4(f"Import finished. Total new jobs added to ChromaDB: {total_imported_count}")
    except Exception as e:
        log_message_step4(f"CRITICAL ERROR during ChromaDB import: {e}")
        traceback.print_exc()
    finally:
        if mongo_client_obj: mongo_client_obj.close()

if __name__ == "__main__":
    main_import_to_chroma()
