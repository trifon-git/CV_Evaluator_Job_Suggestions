import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from datetime import datetime  


load_dotenv()

# Configuration
MODEL_NAME_FOR_EMBEDDING = os.getenv('MODEL_NAME', 'paraphrase-multilingual-mpnet-base-v2') 
OUTPUT_EMBEDDINGS_FILE = None # Will be set dynamically based on input file name

# --- Helper Functions ---
def log_message(message):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {message}")

def generate_average_skill_embedding(model, skills_list):
    """
    Generates an average embedding for a list of skill strings.
    """
    if not skills_list or not isinstance(skills_list, list):
        return None
    
    # Filter out any empty or whitespace-only skills
    valid_skills = [skill for skill in skills_list if isinstance(skill, str) and skill.strip()]
    if not valid_skills:
        return None

    try:
        skill_embeddings = model.encode(valid_skills, show_progress_bar=False)
        if skill_embeddings is None or len(skill_embeddings) == 0:
            return None
        
        average_embedding = np.mean(skill_embeddings, axis=0)
        
        # Normalize the average embedding (optional but often good practice)
        norm = np.linalg.norm(average_embedding)
        if norm > 0:
            average_embedding = average_embedding / norm
            
        return average_embedding.tolist() # Convert to list for JSON serialization
    except Exception as e:
        log_message(f"ERROR generating embeddings for skills: {valid_skills} - {e}")
        return None

def process_extractions_file(input_filepath):
    global OUTPUT_EMBEDDINGS_FILE
    if not os.path.exists(input_filepath):
        log_message(f"ERROR: Input file not found: {input_filepath}")
        return

    # Dynamically create output filename based on input
    input_dir, input_filename_ext = os.path.split(input_filepath)
    input_filename, _ = os.path.splitext(input_filename_ext)
    OUTPUT_EMBEDDINGS_FILE = os.path.join(input_dir, f"{input_filename}_WITH_EMBEDDINGS.jsonl")
    log_message(f"Output will be saved to: {OUTPUT_EMBEDDINGS_FILE}")


    log_message(f"Loading sentence transformer model: {MODEL_NAME_FOR_EMBEDDING}")
    try:
        model = SentenceTransformer(MODEL_NAME_FOR_EMBEDDING)
        log_message("Model loaded successfully.")
    except Exception as e:
        log_message(f"ERROR loading sentence transformer model: {e}")
        return

    processed_records = 0
    embedded_records_count = 0

    # First pass to count lines for tqdm progress bar
    total_lines = 0
    with open(input_filepath, 'r', encoding='utf-8') as infile:
        for _ in infile:
            total_lines += 1
    
    log_message(f"Found {total_lines} records in {input_filepath}")

    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_EMBEDDINGS_FILE, 'w', encoding='utf-8') as outfile:
        
        for line in tqdm(infile, total=total_lines, desc="Processing records"):
            try:
                record = json.loads(line.strip())
                processed_records += 1

                extracted_data = record.get("extracted_data", {})
                skills = extracted_data.get("skills", [])
                
                skill_embedding_vector = None
                if skills:
                    skill_embedding_vector = generate_average_skill_embedding(model, skills)
                
                if skill_embedding_vector:
                    record["skill_embedding"] = skill_embedding_vector
                    embedded_records_count +=1
                else:
                    record["skill_embedding"] = None # Explicitly set to None if no skills or embedding failed

                json.dump(record, outfile, ensure_ascii=False)
                outfile.write('\n')

            except json.JSONDecodeError as e:
                log_message(f"Skipping malformed line: {line.strip()} - Error: {e}")
            except Exception as e:
                log_message(f"Error processing record: {line.strip()} - Error: {e}")
    
    log_message(f"Finished processing. Processed {processed_records} records.")
    log_message(f"Successfully generated embeddings for {embedded_records_count} records.")
    log_message(f"Output saved to: {OUTPUT_EMBEDDINGS_FILE}")


if __name__ == "__main__":
    # --- Find the latest test_llm_chunked_extractions_*.jsonl file ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Assumes this script is in src/
    data_dir = os.path.join(project_root, "data")
    
    latest_extraction_file = None
    latest_timestamp = None

    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.startswith("test_llm_chunked_extractions_") and filename.endswith(".jsonl"):
                try:
                    # Extract timestamp from filename like test_llm_chunked_extractions_YYYYMMDD_HHMMSS.jsonl
                    timestamp_str = filename.replace("test_llm_chunked_extractions_", "").replace(".jsonl", "")
                    file_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    if latest_timestamp is None or file_timestamp > latest_timestamp:
                        latest_timestamp = file_timestamp
                        latest_extraction_file = os.path.join(data_dir, filename)
                except ValueError:
                    continue # Skip files with incorrectly formatted timestamps
    
    if latest_extraction_file:
        log_message(f"Found latest extraction file: {latest_extraction_file}")
        process_extractions_file(latest_extraction_file)
    else:
        log_message(f"ERROR: No 'test_llm_chunked_extractions_*.jsonl' file found in '{data_dir}'.")
        log_message("Please run test_llm.py first to generate an extraction file.")