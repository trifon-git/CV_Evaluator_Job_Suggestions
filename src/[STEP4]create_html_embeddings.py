import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import threading

# Configuration
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "job_listings.db")
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
BATCH_SIZE = 50

# Get script name for log files
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

# Thread-local storage for database connections
local_storage = threading.local()

def log_message(message, is_error=False):
    """Log a message with timestamp to both console and appropriate log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    
    # Print to console
    print(formatted_message)
    
    # Determine log file paths
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Log to script-specific log file
    log_path = os.path.join(logs_dir, f"{SCRIPT_NAME}_log.txt")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"{formatted_message}\n")
    
    # Additionally log errors to script-specific error log file
    if is_error:
        error_log_path = os.path.join(logs_dir, f"{SCRIPT_NAME}_error_log.txt")
        with open(error_log_path, "a", encoding="utf-8") as error_file:
            error_file.write(f"{formatted_message}\n")

class EmbeddingModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.vector_dimension = None
        self.max_seq_length = None
        self.chunk_size = None
        self.chunk_overlap = None
        
    def load(self):
        log_message(f"Loading embedding model: {self.model_name}...")
        try:
            self.model = SentenceTransformer(self.model_name)
            self.vector_dimension = self.model.get_sentence_embedding_dimension()
            self.max_seq_length = self.model.max_seq_length
            self.tokenizer = self.model.tokenizer
            
            self.chunk_size = self.max_seq_length
            self.chunk_overlap = max(50, self.chunk_size // 4)
            
            log_message(f"Model loaded. Vector dimension: {self.vector_dimension}, Max sequence length: {self.max_seq_length}")
            return True
        except Exception as e:
            log_message(f"Failed to load model '{self.model_name}': {e}", is_error=True)
            return False
            
    def get_embedding(self, text):
        if not text or not isinstance(text, str):
            return None
            
        try:
            tokens = self.tokenizer.encode(text, truncation=False, add_special_tokens=False)
            total_tokens = len(tokens)
            
            if total_tokens <= self.max_seq_length:
                embedding = self.model.encode(text, normalize_embeddings=True)
                return embedding

            # For longer texts, chunk by tokens
            chunk_embeddings = []
            start_index = 0
            step = self.chunk_size - self.chunk_overlap
            
            while start_index < total_tokens:
                end_index = min(start_index + self.max_seq_length, total_tokens)
                chunk_token_ids = tokens[start_index:end_index]
                chunk_text = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
                
                if not chunk_text.strip():
                    start_index += step
                    continue
                    
                chunk_embedding = self.model.encode(chunk_text, normalize_embeddings=True)
                chunk_embeddings.append(chunk_embedding)
                start_index += step

            if not chunk_embeddings:
                return None
                
            average_embedding = np.mean(chunk_embeddings, axis=0)
            norm = np.linalg.norm(average_embedding)
            
            if norm == 0:
                return average_embedding
                
            return average_embedding / norm
            
        except Exception as e:
            log_message(f"Error during embedding generation: {e}", is_error=True)
            return None

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn_lock = threading.RLock()
        
    def get_connection(self):
        if not hasattr(local_storage, 'db_conn'):
            with self.conn_lock:
                try:
                    conn = sqlite3.connect(self.db_path, timeout=30)
                    conn.row_factory = sqlite3.Row
                    local_storage.db_conn = conn
                    log_message(f"Thread {threading.get_ident()}: Created new database connection")
                except Exception as e:
                    log_message(f"Thread {threading.get_ident()}: Failed to create database connection: {e}")
                    raise
        
        return local_storage.db_conn
    
    def close(self):
        with self.conn_lock:
            if hasattr(local_storage, 'db_conn'):
                local_storage.db_conn.close()
                delattr(local_storage, 'db_conn')

def vector_to_blob(vector):
    """Convert numpy array to binary blob for storage."""
    return np.array(vector, dtype=np.float32).tobytes() if vector is not None else None

def process_jobs_batch(model, db_manager, batch):
    """Process a batch of jobs and update their embeddings."""
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    
    for job_id, html_content in batch:
        try:
            embedding = model.get_embedding(html_content)
            if embedding is not None:
                blob = vector_to_blob(embedding)
                cursor.execute('''
                    UPDATE jobs 
                    SET embedding = ?
                    WHERE id = ?
                ''', (blob, job_id))
                log_message(f"Updated embedding for job ID: {job_id}")
            else:
                log_message(f"Could not generate embedding for job ID: {job_id}")
        except Exception as e:
            log_message(f"Error processing job ID {job_id}: {e}")
            continue
    
    conn.commit()

def ensure_columns(conn):
    """Ensure required columns exist in the jobs table."""
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(jobs)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        if 'embedding' not in column_names:
            cursor.execute('ALTER TABLE jobs ADD COLUMN embedding BLOB')
            conn.commit()
            log_message("Successfully added embedding column to jobs table")
        else:
            log_message("Embedding column already exists in jobs table")
            
    except sqlite3.Error as e:
        log_message(f"Database error while checking/creating columns: {e}", is_error=True)
        raise

def extract_posting_date(html_content):
    """Extract posting date from HTML content.
    You'll need to implement this based on your HTML structure.
    """
    try:
        # Example implementation - modify based on your HTML structure
        # This is a placeholder - implement actual date extraction logic
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        date_element = soup.find('div', class_='jobposting-date')  # Adjust selector based on your HTML
        if date_element:
            return date_element.text.strip()
        return None
    except Exception as e:
        log_message(f"Error extracting posting date: {e}", is_error=True)
        return None

def main():
    log_message("Starting HTML content embedding generation process")
    
    # Initialize model and database manager
    model = EmbeddingModel(MODEL_NAME)
    if not model.load():
        log_message("Failed to load embedding model. Exiting.")
        return
        
    db_manager = DatabaseManager(DB_PATH)
    
    try:
        conn = db_manager.get_connection()
        # Ensure required columns exist
        ensure_columns(conn)
        
        cursor = conn.cursor()
        
        # Get total count of jobs needing processing
        cursor.execute('''
            SELECT COUNT(*) 
            FROM jobs 
            WHERE html_content IS NOT NULL 
            AND embedding IS NULL
        ''')
        total_jobs = cursor.fetchone()[0]
        log_message(f"Found {total_jobs} jobs needing embeddings")
        
        processed = 0
        while True:
            # Get next batch of jobs
            cursor.execute('''
                SELECT id, html_content 
                FROM jobs 
                WHERE html_content IS NOT NULL 
                AND embedding IS NULL
                LIMIT ?
            ''', (BATCH_SIZE,))
            
            batch = cursor.fetchall()
            if not batch:
                break
                
            process_jobs_batch(model, db_manager, batch)
            processed += len(batch)
            
            if processed % 50 == 0:
                log_message(f"Progress: {processed}/{total_jobs} jobs processed")
        
        log_message(f"Completed! Total jobs processed: {processed}")
        
    except Exception as e:
        log_message(f"Unexpected error: {e}", is_error=True)
    finally:
        db_manager.close()

if __name__ == "__main__":
    main()