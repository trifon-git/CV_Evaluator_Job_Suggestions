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

    # The important part is here - all skills are averaged into ONE embedding vector
    average_embedding = np.mean(skill_embeddings_np, axis=0)
    norm = np.linalg.norm(average_embedding)
    
    return (average_embedding / norm if norm > 0 else average_embedding).tolist()

# New function for direct skill-to-skill matching
def calculate_enhanced_job_match_score(cv_skills, job_skills, cv_skill_embeddings_dict, job_embedding):
    """
    Enhanced scoring that considers:
    1. Direct skill matches (exact or near matches)
    2. Semantic similarity between individual CV skills and the job
    3. Overall semantic similarity between combined CV and job embeddings
    4. Contextual importance of skills in the job description
    
    Args:
        cv_skills: List of skills from CV
        job_skills: List of skills from job ad
        cv_skill_embeddings_dict: Dictionary mapping CV skills to their embeddings
        job_embedding: Job embedding vector from ChromaDB
    
    Returns:
        tuple: (enhanced_score, match_details)
    """
    if not cv_skills or not job_skills or not cv_skill_embeddings_dict or job_embedding is None:
        return 0.0, []
    
    # 1. Calculate direct skill matches (exact or near matches)
    cv_skills_lower = [s.lower().strip() for s in cv_skills if isinstance(s, str)]
    job_skills_lower = [s.lower().strip() for s in job_skills if isinstance(s, str)]
    
    exact_matches = []
    for cv_skill in cv_skills_lower:
        for job_skill in job_skills_lower:
            # Check for exact match
            if cv_skill == job_skill:
                exact_matches.append((cv_skill, job_skill, 1.0))
            # Check for substring (partial match)
            elif cv_skill in job_skill or job_skill in cv_skill:
                # Shorter text length ratio to avoid spurious substring matches
                len_ratio = min(len(cv_skill), len(job_skill)) / max(len(cv_skill), len(job_skill))
                if len_ratio >= 0.7:  # Only consider substantial overlaps
                    exact_matches.append((cv_skill, job_skill, 0.9 * len_ratio))
    
    # 2. Calculate semantic similarity for each CV skill to the job embedding
    skill_similarities = []
    job_emb_np = np.array(job_embedding)
    
    for skill in cv_skills:
        if skill in cv_skill_embeddings_dict:
            skill_vec = cv_skill_embeddings_dict[skill]
            sim = cosine_similarity_np(skill_vec, job_emb_np)
            skill_similarities.append((skill, sim))
    
    # 3. Calculate bidirectional alignment score - how well each job skill aligns with CV skills
    job_skill_alignment_scores = []
    if job_skills and cv_skill_embeddings_dict:
        for job_skill in job_skills:
            # Create a simple embedding for the job skill using averaged CV skills
            best_alignment = 0
            for cv_skill, cv_emb in cv_skill_embeddings_dict.items():
                # Simple text similarity check
                if job_skill.lower() in cv_skill.lower() or cv_skill.lower() in job_skill.lower():
                    alignment = 0.8  # High alignment for text overlap
                else:
                    # Use semantic similarity when available
                    alignment = cosine_similarity_np(cv_emb, job_emb_np) * 0.5  # Weighted lower
                best_alignment = max(best_alignment, alignment)
            job_skill_alignment_scores.append((job_skill, best_alignment))
    
    # 4. Compute the enhanced score with all components
    num_job_skills = max(1, len(job_skills_lower))
    
    # Exact match component (weighted by % of job skills matched)
    exact_match_score = sum(score for _, _, score in exact_matches) / num_job_skills
    
    # Semantic similarity component (from CV to job)
    if skill_similarities:
        # Use top matches - focus on best skills
        skill_similarities.sort(key=lambda x: x[1], reverse=True)
        top_count = max(1, min(5, len(skill_similarities)))
        semantic_score = sum(sim for _, sim in skill_similarities[:top_count]) / top_count
    else:
        semantic_score = 0
    
    # Bidirectional alignment component (from job to CV)
    if job_skill_alignment_scores:
        # Weight critical/required job skills higher (if we had that info)
        job_alignment_score = sum(score for _, score in job_skill_alignment_scores) / len(job_skill_alignment_scores)
    else:
        job_alignment_score = 0
    
    # Combined score with adjusted weights
    # - Direct matches are most important (50%)
    # - Semantic similarity is next (30%)
    # - Bidirectional alignment provides additional context (20%)
    enhanced_score = (exact_match_score * 0.5) + (semantic_score * 0.3) + (job_alignment_score * 0.2)
    
    # Scale to 0-100
    final_score = min(100, enhanced_score * 100)
    
    # Prepare match details for explanation
    match_details = []
    
    # Add exact matches to details
    for cv_skill, job_skill, match_score in exact_matches:
        match_details.append({
            'cv_skill': cv_skill,
            'job_skill': job_skill,
            'match_type': 'exact' if match_score == 1.0 else 'partial',
            'score': match_score
        })
    
    # Add top semantic matches to details
    if skill_similarities:
        for skill, sim in skill_similarities[:5]:  # Take top 5 for details
            if not any(d['cv_skill'] == skill for d in match_details):  # Avoid duplicates
                match_details.append({
                    'cv_skill': skill,
                    'job_skill': None,  # No specific job skill - semantic match
                    'match_type': 'semantic',
                    'score': sim
                })
    
    # Add bidirectional alignment details
    if job_skill_alignment_scores:
        job_skill_alignment_scores.sort(key=lambda x: x[1], reverse=True)
        for job_skill, align_score in job_skill_alignment_scores[:3]:  # Take top 3
            match_details.append({
                'cv_skill': None,  # No specific CV skill - alignment with overall profile
                'job_skill': job_skill,
                'match_type': 'alignment',
                'score': align_score
            })
    
    return final_score, match_details

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

def explain_job(cv_skills, job_skills_from_meta, cv_skill_embeddings_dict, job_embedding_from_chroma):
    """
    Optimized version that uses pre-calculated CV skill embeddings.
    
    Args:
        cv_skills: List of CV skill strings
        job_skills_from_meta: List of job skill strings (not used in current implementation)
        cv_skill_embeddings_dict: Dict mapping CV skill strings to their embeddings
        job_embedding_from_chroma: Job embedding vector from ChromaDB
    """
    # Fix the boolean check for numpy arrays
    job_embedding_valid = (
        job_embedding_from_chroma is not None and 
        hasattr(job_embedding_from_chroma, '__len__') and 
        len(job_embedding_from_chroma) > 0
    )
    
    if not cv_skills or not job_embedding_valid or not cv_skill_embeddings_dict:
        return []

    skill_contributions = []
    job_emb_np = np.array(job_embedding_from_chroma)

    # This calculates similarity for EACH individual skill to explain matches
    for skill_text in cv_skills:
        if skill_text in cv_skill_embeddings_dict:
            skill_vector = cv_skill_embeddings_dict[skill_text]
            similarity = cosine_similarity_np(skill_vector, job_emb_np)
            skill_contributions.append((skill_text, similarity))

    if not skill_contributions:
        print("No CV skills had pre-calculated embeddings for explanation.")
        return []

    # Sort by similarity and return top N
    skill_contributions.sort(key=lambda x: x[1], reverse=True)
    return skill_contributions[:EXPLAIN_TOP_N_CONTRIBUTING_SKILLS]

def find_similar_jobs(cv_skills, cv_embedding, top_n=TOP_N_RESULTS_DEFAULT, filter_active_only=True, use_enhanced_scoring=True):
    if not cv_skills or cv_embedding is None:
        print("Error: CV skills or embedding is missing.")
        return []

    try:
        # Pre-calculate individual CV skill embeddings ONCE
        print(f"Pre-calculating embeddings for {len(cv_skills)} CV skills...")
        cv_individual_skill_embeddings = {}  # Dict to store skill -> embedding mapping
        
        valid_cv_skills = [s for s in cv_skills if isinstance(s, str) and s.strip()]
        if valid_cv_skills:
            if EMBEDDING_API_URL:
                cv_skill_vectors = get_remote_embedding_batch(valid_cv_skills)
            else:
                model = get_embedding_model()
                if model:
                    try:
                        raw_embeddings = model.encode(valid_cv_skills, show_progress_bar=False)
                        cv_skill_vectors = [np.array(e) for e in raw_embeddings]
                    except Exception as e:
                        print(f"Error embedding CV skills: {e}")
                        cv_skill_vectors = [None] * len(valid_cv_skills)
                else:
                    cv_skill_vectors = [None] * len(valid_cv_skills)
            
            # Store embeddings in dictionary for quick lookup
            for i, skill in enumerate(valid_cv_skills):
                if i < len(cv_skill_vectors) and cv_skill_vectors[i] is not None:
                    cv_individual_skill_embeddings[skill] = cv_skill_vectors[i]
        
        print(f"Successfully pre-calculated embeddings for {len(cv_individual_skill_embeddings)} CV skills.")

        # Connect to ChromaDB
        chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=ChromaSettings(allow_reset=True))
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        print(f"Connected to ChromaDB collection: {COLLECTION_NAME}")

        # Query ChromaDB
        query_filter = {"Status": "active"} if filter_active_only else None
        if query_filter:
            print("Filtering for active jobs only in ChromaDB.")

        print(f"Querying ChromaDB with top_n={top_n}...")
        
        # Convert cv_embedding to the right format for ChromaDB
        if isinstance(cv_embedding, np.ndarray):
            query_embedding_list = cv_embedding.tolist()
        elif isinstance(cv_embedding, list):
            query_embedding_list = cv_embedding
        else:
            print(f"Warning: Unexpected cv_embedding type: {type(cv_embedding)}")
            query_embedding_list = list(cv_embedding)

        # The main search uses the COMBINED embedding to find similar jobs
        results = collection.query(
            query_embeddings=[query_embedding_list],
            n_results=top_n,
            where=query_filter,
            include=["metadatas", "distances", "documents", "embeddings"]
        )

        if not results or not results['ids'] or not results['ids'][0]:
            print("No results returned from ChromaDB.")
            return []

        print(f"ChromaDB returned {len(results['ids'][0])} raw results.")

        # Process results
        matches = []
        for i in range(len(results['ids'][0])):
            try:
                metadata = results['metadatas'][0][i] if results.get('metadatas') and len(results['metadatas'][0]) > i else {}
                distance = results['distances'][0][i] if results.get('distances') and len(results['distances'][0]) > i else 1.0
                document = results['documents'][0][i] if results.get('documents') and len(results['documents'][0]) > i else ""
                
                # Fix the numpy array boolean evaluation issue - CONSOLIDATED DEFINITION
                job_embedding = None
                if results.get('embeddings') and results['embeddings'] and len(results['embeddings']) > 0 and len(results['embeddings'][0]) > i:
                    job_embedding = results['embeddings'][0][i]
                
                # Check job_embedding validity ONCE
                job_embedding_valid = (
                    job_embedding is not None and 
                    hasattr(job_embedding, '__len__') and 
                    len(job_embedding) > 0
                )

                # Convert distance to a percentage score (closer = higher score)
                score = max(0, min(100, (1 - distance) * 100))
                
                # Get job skills from metadata
                job_skills_list = []
                skills_json_str = metadata.get('Skills_Json_Str', '[]')
                try:
                    job_skills_list = json.loads(skills_json_str) if skills_json_str else []
                except (json.JSONDecodeError, TypeError):
                    job_skills_list = []

                # Calculate score with enhanced method if requested
                match_details = []
                if use_enhanced_scoring and job_embedding_valid and job_skills_list and cv_individual_skill_embeddings:
                    # Use the new enhanced scoring method
                    enhanced_score, match_details = calculate_enhanced_job_match_score(
                        cv_skills, 
                        job_skills_list,
                        cv_individual_skill_embeddings,
                        job_embedding
                    )
                    # Blend with original vector similarity score for stability
                    original_score = max(0, min(100, (1 - distance) * 100))
                    score = (enhanced_score * 0.7) + (original_score * 0.3)

                # Use pre-calculated embeddings for explanation
                contributing_cv_skills = []
                if job_embedding_valid and cv_individual_skill_embeddings:
                    contributing_cv_skills = explain_job(
                        cv_skills, 
                        job_skills_list, 
                        cv_individual_skill_embeddings,
                        job_embedding
                    )

                # Use the correct field names as confirmed by debug output
                title = metadata.get('Title', 'N/A')
                company = metadata.get('Company', 'N/A')
                area = metadata.get('Area', 'N/A')
                category = metadata.get('Category', 'N/A')
                url = metadata.get('URL', '')

                # Prepare the match entry
                match = {
                    'job_id': results['ids'][0][i],
                    'score': score,
                    'title': title,
                    'company': company,
                    'area': area,
                    'category': category,
                    'url': url,
                    'document': document,
                    'job_skills': job_skills_list,
                    'contributing_skills': contributing_cv_skills,
                    'match_details': match_details,
                    'metadata': metadata
                }
                matches.append(match)

            except Exception as e:
                print(f"Error processing match {i}: {e}")
                traceback.print_exc()  # Add traceback for better debugging
                continue

        print(f"Processed {len(matches)} matches from ChromaDB results.")
        return matches

    except Exception as e:
        print(f"Error in find_similar_jobs: {e}")
        traceback.print_exc()
        return []
    


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
            try:
                job_matches_found = find_similar_jobs(cv_skills=sample_cv_skills_list, cv_embedding=test_cv_skill_embedding, top_n=5)
                if job_matches_found:
                    print(f"Found {len(job_matches_found)} job matches:")
                    for i_match, match_item in enumerate(job_matches_found):
                        print(f"\n  {i_match+1}. Title: {match_item.get('title', 'N/A')}")  # Use lowercase field name
                        print(f"     Score: {match_item.get('score', 0.0):.2f}%")
                        print(f"     Company: {match_item.get('company', 'N/A')}, Area: {match_item.get('area', 'N/A')}")
                        print(f"     Category: {match_item.get('category', 'N/A')}")  # Test: print Category
                        print(f"     Contributing CV Skills: {[(s_item[0], round(s_item[1], 3)) for s_item in match_item.get('contributing_skills', [])]}")
                else:
                    print("No job matches found for the test CV skills.")
            except Exception as e:
                print(f"Error during job matching test: {e}")
                traceback.print_exc()
        else:
            print("Failed to generate embedding for test CV skills.")