import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime # For finding the latest file

# --- Configuration ---
load_dotenv()
MODEL_NAME_FOR_EMBEDDING = os.getenv('MODEL_NAME', 'paraphrase-multilingual-mpnet-base-v2')
TOP_N_MATCHES_TO_SHOW = 5

# --- Helper Functions ---
def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {message}")

def generate_average_skill_embedding(model, skills_list):
    if not skills_list or not isinstance(skills_list, list): return None
    valid_skills = [skill for skill in skills_list if isinstance(skill, str) and skill.strip()]
    if not valid_skills: return None
    try:
        skill_embeddings = model.encode(valid_skills, show_progress_bar=False)
        if skill_embeddings is None or len(skill_embeddings) == 0: return None
        average_embedding = np.mean(skill_embeddings, axis=0)
        norm = np.linalg.norm(average_embedding)
        if norm > 0: average_embedding = average_embedding / norm
        return average_embedding # Return as numpy array for direct cosine similarity calculation
    except Exception as e:
        log_message(f"ERROR generating embeddings for skills: {valid_skills} - {e}")
        return None

def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None: return 0.0
    # Ensure they are numpy arrays (they should be from generate_average_skill_embedding,
    # but skill_embedding from file is a list)
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

def load_job_data_with_embeddings(input_filepath):
    if not os.path.exists(input_filepath):
        log_message(f"ERROR: Embeddings file not found: {input_filepath}")
        return []
    
    job_ads_data = []
    with open(input_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                # Ensure essential fields are present for matching
                if "source_details" in record and "extracted_data" in record and "skill_embedding" in record:
                    if record["skill_embedding"] is not None: # Only consider jobs with embeddings
                        job_ads_data.append(record)
            except json.JSONDecodeError:
                log_message(f"Skipping malformed line in embeddings file: {line.strip()}")
    log_message(f"Loaded {len(job_ads_data)} job records with embeddings from {input_filepath}")
    return job_ads_data

# --- Main Test Logic ---
if __name__ == "__main__":
    log_message("--- Skill Matching Test Started ---")

    # 1. Find the latest _WITH_EMBEDDINGS.jsonl file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    
    latest_embeddings_file = None
    latest_timestamp_obj = None

    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.startswith("test_llm_chunked_extractions_") and filename.endswith("_WITH_EMBEDDINGS.jsonl"):
                try:
                    # Extract timestamp, e.g., from "test_llm_chunked_extractions_YYYYMMDD_HHMMSS_WITH_EMBEDDINGS.jsonl"
                    base_name = filename.replace("_WITH_EMBEDDINGS.jsonl", "")
                    timestamp_str = base_name.replace("test_llm_chunked_extractions_", "")
                    file_timestamp_obj = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    if latest_timestamp_obj is None or file_timestamp_obj > latest_timestamp_obj:
                        latest_timestamp_obj = file_timestamp_obj
                        latest_embeddings_file = os.path.join(data_dir, filename)
                except ValueError:
                    log_message(f"Warning: Could not parse timestamp from filename: {filename}")
                    continue
    
    if not latest_embeddings_file:
        log_message(f"ERROR: No suitable '*_WITH_EMBEDDINGS.jsonl' file found in '{data_dir}'.")
        log_message("Please run '[STEP_EMBED]embed_extracted_skills.py' first.")
        sys.exit(1)
    
    log_message(f"Using embeddings file: {latest_embeddings_file}")
    
    # 2. Load Job Data with Embeddings
    job_ads = load_job_data_with_embeddings(latest_embeddings_file)
    if not job_ads:
        log_message("No job ads loaded. Exiting.")
        sys.exit(1)

    # 3. Load Embedding Model (same one used to create the job skill embeddings)
    log_message(f"Loading sentence transformer model for CV skills: {MODEL_NAME_FOR_EMBEDDING}")
    try:
        embedding_model = SentenceTransformer(MODEL_NAME_FOR_EMBEDDING)
        log_message("Embedding model loaded successfully.")
    except Exception as e:
        log_message(f"ERROR loading sentence transformer model: {e}")
        sys.exit(1)

    # 4. Define Sample CV Skill Sets
    sample_cvs = [
        {
            "cv_id": "CV_Python_Django_Dev",
            "skills": ["Python", "Django", "PostgreSQL", "RESTful APIs", "Problem-solving", "Git"]
        },
        {
            "cv_id": "CV_Embedded_C_Linux_Dev",
            "skills": ["C++", "Embedded Linux", "Low level processor and hardware knowledge", "RTOS", "Debugging", "System understanding"]
        },
        {
            "cv_id": "CV_Frontend_React_Dev",
            "skills": ["JavaScript", "React", "HTML5", "CSS3", "Node.js", "UI/UX Design Principles", "Agile methodologies"]
        },
        { # A CV that might match the Danish job ad well
            "cv_id": "CV_Mainframe_Modernization_Dev",
            "skills": ["COBOL", "REXX", "Java", "TypeScript", "Mainframe Development", "CI/CD", "Jenkins", "Git", "Agile mindset", "Problem-solving"]
        }
    ]

    # 5. Perform Matching for each CV
    for cv in sample_cvs:
        log_message(f"\n--- Matching for: {cv['cv_id']} ---")
        log_message(f"CV Skills: {cv['skills']}")
        
        cv_skill_vector = generate_average_skill_embedding(embedding_model, cv['skills'])
        
        if cv_skill_vector is None:
            log_message("Could not generate embedding for CV skills. Skipping this CV.")
            continue

        matches = []
        for job_ad in job_ads:
            job_ad_skill_vector = job_ad.get("skill_embedding")
            if job_ad_skill_vector:
                similarity = cosine_similarity(cv_skill_vector, np.array(job_ad_skill_vector)) # Ensure job vector is numpy array
                
                source_info = job_ad.get("source_details", {})
                job_title = source_info.get("job_title", "Unknown Title")
                mongo_id = source_info.get("mongo_id", "N/A")
                job_skills = job_ad.get("extracted_data", {}).get("skills", [])

                matches.append({
                    "job_title": job_title,
                    "mongo_id": mongo_id,
                    "similarity": similarity,
                    "job_skills_count": len(job_skills)
                    # "job_skills_preview": job_skills[:5] # Optionally add a preview
                })
        
        # Sort matches by similarity (descending)
        sorted_matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)
        
        log_message(f"Top {TOP_N_MATCHES_TO_SHOW} potential matches:")
        if not sorted_matches:
            log_message("No matches found (or no jobs with embeddings).")
        for i, match in enumerate(sorted_matches[:TOP_N_MATCHES_TO_SHOW]):
            log_message(
                f"  {i+1}. \"{match['job_title']}\" (ID: {match['mongo_id']}) - "
                f"Similarity: {match['similarity']:.4f} (Job Skills: {match['job_skills_count']})"
            )
    
    log_message("\n--- Skill Matching Test Finished ---")