import os
import json
import sys
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load environment variables
load_dotenv()

MODEL_NAME_FOR_EMBEDDING = os.getenv('MODEL_NAME', 'paraphrase-multilingual-mpnet-base-v2')
TOP_N_MATCHES_TO_SHOW = 5  
EXPLAIN_TOP_N_CONTRIBUTING_SKILLS = 3  # How many contributing CV skills to show

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
    
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm_vec1, norm_vec2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    
    # Avoid division by zero
    return 0.0 if norm_vec1 == 0 or norm_vec2 == 0 else dot_product / (norm_vec1 * norm_vec2)

def load_job_data_with_embeddings(filepath):
    if not os.path.exists(filepath):
        log_message(f"Can't find embeddings file: {filepath}")
        return []
        
    job_ads_data = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                job_ad_record = json.loads(line.strip())
                
                # Make sure we have all required fields
                required_fields = ["source_details", "extracted_data", "skill_embedding"]
                if all(k in job_ad_record for k in required_fields) and job_ad_record["skill_embedding"]:
                    job_ads_data.append(job_ad_record)
            except json.JSONDecodeError:
                log_message(f"Skipping bad line: {line[:50]}...")
                
    log_message(f"Got {len(job_ads_data)} usable job records from {filepath}")
    return job_ads_data

# ----------------------
# Main script execution
# ----------------------
if __name__ == "__main__":
    log_message("--- Skill Matching Test Started ---")

    # Find the project root and data directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    
    # Find the latest embeddings file
    latest_embeddings_file = None
    latest_timestamp_obj = None

    if os.path.exists(data_dir):
        for fname in os.listdir(data_dir):
            # Look for files with the right naming pattern
            if fname.startswith("test_llm_chunked_extractions_") and fname.endswith("_WITH_EMBEDDINGS.jsonl"):
                try:
                    # Extract timestamp from filename
                    base = fname.replace("_WITH_EMBEDDINGS.jsonl", "")
                    ts_str = base.replace("test_llm_chunked_extractions_", "")
                    file_time_obj = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                    
                    # Keep track of the most recent file
                    if latest_timestamp_obj is None or file_time_obj > latest_timestamp_obj:
                        latest_timestamp_obj = file_time_obj
                        latest_embeddings_file = os.path.join(data_dir, fname)
                except ValueError:
                    log_message(f"Weird filename format, ignoring: {fname}")
                    continue
    
    # Bail out if we can't find any embeddings files
    if not latest_embeddings_file:
        log_message(f"No embeddings files in '{data_dir}'. Run [STEP_EMBED]embed_extracted_skills.py first.")
        sys.exit(1)
    
    log_message(f"Using embeddings file: {latest_embeddings_file}")
    
    # Load job data
    job_ads = load_job_data_with_embeddings(latest_embeddings_file)
    if not job_ads:
        log_message("No job ads loaded. Bailing out.")
        sys.exit(1)

    # Load the embedding model
    log_message(f"Loading embedding model: {MODEL_NAME_FOR_EMBEDDING}")
    try:
        embedding_model = SentenceTransformer(MODEL_NAME_FOR_EMBEDDING)
        log_message("Embedding Model loaded.")
    except Exception as e:
        log_message(f"Failed to load embedding model: {e}")
        sys.exit(1)

    # Define some sample CVs for testing
    sample_cvs = [
        {
            "cv_id": "CV_Python_Django_Dev",
            "skills": ["Python", "Django", "PostgreSQL", "RESTful APIs", "Problem-solving", "Git", "Software Development"]
        },
        {
            "cv_id": "CV_Embedded_C_Linux_Dev",
            "skills": ["C++", "Embedded Linux", "Low level processor and hardware knowledge", "RTOS", "Debugging", "System understanding", "Firmware"]
        },
        {
            "cv_id": "CV_Frontend_React_Dev",
            "skills": ["JavaScript", "React", "HTML5", "CSS3", "Node.js", "UI/UX Design Principles", "Agile methodologies", "Responsive Design"]
        },
        { 
            "cv_id": "CV_Mainframe_Modernization_Dev",
            "skills": ["COBOL", "REXX", "Java", "TypeScript", "Mainframe Development", "CI/CD", "Jenkins", "Git", "Agile mindset", "Problem-solving", "SQL"]
        }
    ]

    # Process each CV
    for cv_data in sample_cvs:
        log_message(f"\n--- Matching CV: {cv_data['cv_id']} ---")
        log_message(f"CV Skills: {cv_data['skills']}")
        
        # Generate embedding for CV skills
        cv_skill_vector = generate_average_skill_embedding(embedding_model, cv_data['skills'])
        
        if cv_skill_vector is None:
            log_message("Couldn't create embedding for these CV skills. Skipping.")
            continue

        # Match against all job ads
        match_results = []
        for job_ad_record in job_ads:
            job_ad_skill_vector = job_ad_record.get("skill_embedding")
            
            if job_ad_skill_vector:
                # Calculate similarity
                similarity_score = cosine_similarity(cv_skill_vector, np.array(job_ad_skill_vector))
                
                # Extract job details
                source_details = job_ad_record.get("source_details", {})
                job_title = source_details.get("job_title", "Unknown Job")
                mongo_id = source_details.get("mongo_id", "N/A")
                job_ad_skills = job_ad_record.get("extracted_data", {}).get("skills", [])

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
                f"(CV Skills: {len(cv_data['skills'])}, Job Skills: {match['job_skills_count']})"
            )

            # --- EXPLAINABILITY ---
            # Show which CV skills contributed most to this match
            job_ad_for_explain = match["job_ad_record_for_explain"]
            job_ad_summary_vector_for_explain = np.array(job_ad_for_explain.get("skill_embedding"))

            log_message("     Key CV skills contributing to this match (similarity to job's overall skill profile):")
            contributing_cv_skills = []
            
            if cv_data['skills']:
                # Filter out empty/None skills from CV before encoding
                valid_cv_skills_for_explain = [s for s in cv_data['skills'] if isinstance(s, str) and s.strip()]
                
                if valid_cv_skills_for_explain:
                    # Get embeddings for each individual skill
                    cv_individual_skill_vectors = embedding_model.encode(valid_cv_skills_for_explain, show_progress_bar=False)
                    
                    # Compare each skill to the job's overall profile
                    for cv_skill_text, cv_skill_vec in zip(valid_cv_skills_for_explain, cv_individual_skill_vectors):
                        sim_to_job_profile = cosine_similarity(cv_skill_vec, job_ad_summary_vector_for_explain)
                        
                        # Only include skills with decent similarity
                        if sim_to_job_profile > 0.4:  # Threshold for relevance
                            contributing_cv_skills.append({
                                "skill": cv_skill_text, 
                                "similarity_to_job_profile": sim_to_job_profile
                            })
                    
                    # Sort by similarity and show top contributors
                    contributing_cv_skills.sort(key=lambda x: x["similarity_to_job_profile"], reverse=True)
                    
                    for item in contributing_cv_skills[:EXPLAIN_TOP_N_CONTRIBUTING_SKILLS]:
                        log_message(f"       - \"{item['skill']}\" (Sim. to Job Profile: {item['similarity_to_job_profile']:.3f})")
                    
                    if not contributing_cv_skills:
                        log_message("       (No individual CV skills strongly aligned with the job's overall profile above threshold 0.4)")
                else:
                    log_message("       (CV has no valid skills to explain contribution)")
            else:
                log_message("       (CV has no skills listed)")
    
    log_message("\n--- Skill Matching Test Finished ---")
