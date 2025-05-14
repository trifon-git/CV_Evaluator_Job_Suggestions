import numpy as np
import time
import os
import traceback
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from chromadb import HttpClient
import requests
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
from tkinter.messagebox import showerror, showinfo
import PyPDF2
import docx
import markdown
from dotenv import load_dotenv, find_dotenv
import importlib.util
import sys

# Dynamically import [STEP3]llm_skill_extractor.py
skill_extractor_path = os.path.join(os.path.dirname(__file__), '[STEP3]llm_skill_extractor.py')
spec = importlib.util.spec_from_file_location('skill_extractor', skill_extractor_path)
skill_extractor = importlib.util.module_from_spec(spec)
sys.modules['skill_extractor'] = skill_extractor
spec.loader.exec_module(skill_extractor)
extract_skills_with_llm = skill_extractor.extract_skills_with_llm

# Reset environment variables
load_dotenv(find_dotenv(), override=True)

# Try to use scipy for faster cosine calculations if available
try:
    from scipy.spatial.distance import cosine as scipy_cosine
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Using NumPy for cosine similarity calculations.")
    print("Install scipy for better performance: pip install scipy")

# Configuration from environment variables
MODEL_NAME = os.getenv('MODEL_NAME', 'paraphrase-multilingual-mpnet-base-v2')
# Configuration from environment variables
TOP_N_RESULTS = int(os.getenv('TOP_N_RESULTS', '10'))
CV_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", os.getenv('CV_FILE_PATH'))
CHROMA_HOST = os.getenv('CHROMA_HOST')
CHROMA_PORT = int(os.getenv('CHROMA_PORT'))
COLLECTION_NAME = os.getenv('CHROMA_COLLECTION')
# Remote embedding API configuration
EMBEDDING_API_URL = os.getenv('EMBEDDING_API_URL', '')
VERIFY_SSL = os.getenv('VERIFY_SSL', 'true').lower() == 'true'

# Add these imports at the top of the file
import urllib3

# Disable SSL warnings if VERIFY_SSL is set to false
if os.getenv('VERIFY_SSL', 'true').lower() != 'true':
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    if vec1 is None or vec2 is None or not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
        print(f"Warning: Invalid vectors for similarity calculation")
        return 0.0
    
    if vec1.shape != vec2.shape:
        print(f"Shape mismatch: {vec1.shape} vs {vec2.shape}")
        return 0.0

    if HAS_SCIPY:
        try:
            similarity = 1 - scipy_cosine(vec1, vec2)
            return similarity if not np.isnan(similarity) else 0.0
        except Exception as e:
            print(f"Scipy calculation failed: {e}. Falling back to NumPy.")
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    dot_product = np.dot(vec1, vec2)
    similarity = dot_product / (norm1 * norm2)
    
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
        
        if total_tokens <= max_seq_length:
            print("Text within model limits, encoding directly")
            return model.encode(cv_text, normalize_embeddings=True)
            
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
        
        avg_embedding = np.mean(chunk_embeddings, axis=0)
        
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
            
        print(f"Embedding generated in {time.time() - start_time:.2f}s")
        return avg_embedding
        
    except Exception as e:
        print(f"Error generating embedding: {e}")
        traceback.print_exc()
        return None

def get_remote_embedding(texts):
    """Call remote API to get embeddings for texts."""
    try:
        print("Calling remote embedding API...")  # Removed URL from print
        response = requests.post(EMBEDDING_API_URL, json={"texts": texts}, verify=VERIFY_SSL)
        response.raise_for_status()
        embeddings = response.json().get("embeddings", [])
        if not embeddings:
            print("Warning: Empty embedding response from API")
            return []
        return embeddings
    except Exception as e:
        print(f"Error calling embedding API: {str(e)}")  # Changed to not show URL in error
        traceback.print_exc()
        return []

def generate_cv_embedding_remote(cv_text):
    """Generate embeddings for CV text using remote API with chunking for long texts."""
    print("Generating CV embedding via remote API...")
    start_time = time.time()
    
    if not cv_text or not isinstance(cv_text, str):
        print("Error: Invalid CV text")
        return None
        
    try:
        # Split text into chunks
        chunk_size = 1000  # characters per chunk
        overlap = 200      # character overlap between chunks

        chunks = []
        start = 0
        while start < len(cv_text):
            end = start + chunk_size
            chunk = cv_text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap  # step with overlap

        print(f"Split CV into {len(chunks)} chunks for remote embedding")

        chunk_embeddings = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            embedding_response = get_remote_embedding([chunk])
            if embedding_response:
                chunk_embeddings.append(embedding_response[0])
            else:
                print("⚠️ Skipped a chunk due to empty embedding response")

        if not chunk_embeddings:
            print("Error: No valid chunks were embedded")
            return None

        # Average the embeddings
        avg_embedding = np.mean(chunk_embeddings, axis=0)
        
        # Normalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
            
        print(f"Remote embedding generated in {time.time() - start_time:.2f}s")
        return avg_embedding
        
    except Exception as e:
        print(f"Error generating remote embedding: {e}")
        traceback.print_exc()
        return None

def generate_skills_embedding_remote(skills_list: list[str]):
    """Generate an averaged embedding for a list of skill strings using remote API."""
    if not skills_list:
        print("Error: No skills provided to generate embedding.")
        return None

    print(f"Generating embedding for {len(skills_list)} skills via remote API...")
    start_time = time.time()

    skill_embeddings = []
    for i, skill in enumerate(skills_list):
        if not skill.strip():
            continue
        print(f"Processing skill {i+1}/{len(skills_list)}: '{skill}'")
        embedding_response = get_remote_embedding([skill]) # get_remote_embedding expects a list
        if embedding_response and len(embedding_response) > 0:
            # Assuming embedding_response[0] is the actual embedding vector
            skill_embeddings.append(np.array(embedding_response[0]))
        else:
            print(f"⚠️ Skipped skill '{skill}' due to empty embedding response")

    if not skill_embeddings:
        print("Error: No valid skill embeddings were generated.")
        return None

    # Average the embeddings
    avg_embedding = np.mean(skill_embeddings, axis=0)
    
    # Normalize
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm
        
    print(f"Skills embedding generated in {time.time() - start_time:.2f}s")
    return avg_embedding

def find_similar_jobs(cv_text, top_n=None, active_only=True):
    """Find jobs similar to the provided CV text based on extracted skills."""
    if top_n is None:
        top_n = TOP_N_RESULTS
    
    # Step 1: Extract skills using LLM (imported function)
    extracted_skills = extract_skills_with_llm(cv_text)
    if not extracted_skills:
        print("Error: Failed to extract skills from CV.")
        return None, "Error: Failed to extract skills from CV"

    # Step 2: Generate embedding from extracted skills
    print("Generating embedding from extracted skills...")
    cv_skill_embedding = generate_skills_embedding_remote(extracted_skills)
        
    if cv_skill_embedding is None:
        return None, "Error: Failed to generate CV skill embedding"
    
    print(f"CV skill embedding shape: {cv_skill_embedding.shape}")
    
    print("Connecting to ChromaDB...")
    try:
        # Update ChromaDB client initialization to use v2 API
        chroma_client = HttpClient(
            host=CHROMA_HOST, 
            port=CHROMA_PORT,
            ssl=False,
            headers={"accept": "application/json", "Content-Type": "application/json"}
        )
        collection = chroma_client.get_collection(COLLECTION_NAME)
        
        search_start = time.time()
        
        # Add filter for active jobs if requested
        where_filter = {"Status": "active"} if active_only else None
        if active_only:
            print("Filtering for active jobs only")
        
        results = collection.query(
            query_embeddings=[cv_skill_embedding.tolist()],  # <-- FIXED: use cv_skill_embedding
            n_results=top_n,
            include=["metadatas", "distances", "documents"],
            where=where_filter
        )
        
        matches = []
        for idx, (metadata, distance, content) in enumerate(zip(
            results['metadatas'][0], 
            results['distances'][0],
            results['documents'][0]
        )):
            # Using exponential decay for more intuitive scoring
            similarity_score = np.exp(-distance) * 100  # Will give scores between 0-100
            matches.append({
                "score": similarity_score,
                "type": "ChromaDB similarity",
                "Title": metadata.get('Title', 'Unknown Position'),  
                "Company": metadata.get('Company', 'N/A'),          
                "Area": metadata.get('Area', 'N/A'),               
                "url": metadata.get('Application_URL', '#'),       
                "posting_date": metadata.get('Published_Date', 'N/A'), 
                "content": content,
                "Status": metadata.get('Status', 'unknown')
            })
        
        print(f"Search completed in {time.time() - search_start:.2f}s")
        return matches, "ChromaDB Vector Search"
        
    except Exception as e:
        print(f"Error during ChromaDB search: {e}")
        traceback.print_exc()
        return None, "Error: ChromaDB search failed"

def create_ui():
    root = tk.Tk()
    root.title("CV Job Matcher")
    root.geometry("800x600")
    
    def select_file():
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("All supported files", "*.txt *.pdf *.doc *.docx *.md"),
                ("Text files", "*.txt"),
                ("PDF files", "*.pdf"),
                ("Word files", "*.doc *.docx"),
                ("Markdown files", "*.md"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            file_entry.delete(0, tk.END)
            file_entry.insert(0, file_path)

    def process_cv():
        file_path = file_entry.get()
        if not file_path:
            showerror("Error", "Please select a CV file first")
            return

        try:
            # Clear previous results
            results_text.delete(1.0, tk.END)
            results_text.insert(tk.END, "Processing CV...\n\n")
            root.update()

            # Read the file based on extension
            cv_text = ""
            file_ext = file_path.lower()

            if file_ext.endswith('.pdf'):
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        cv_text += page.extract_text() + "\n"
            
            elif file_ext.endswith('.docx'):
                doc = docx.Document(file_path)
                cv_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            elif file_ext.endswith('.md'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    md_text = file.read()
                    # Convert markdown to plain text
                    cv_text = markdown.markdown(md_text)
                    # Remove HTML tags
                    cv_text = cv_text.replace('<p>', '').replace('</p>', '\n')
                    cv_text = cv_text.replace('<br>', '\n')
            
            else:  # Handle as text file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        cv_text = f.read()
                except UnicodeDecodeError:
                    with open(file_path, "r", encoding="latin-1") as f:
                        cv_text = f.read()

            if not cv_text.strip():
                showerror("Error", "Could not extract text from file")
                return

            # Check if embedding API URL is set
            if not EMBEDDING_API_URL:
                showerror("Error", "EMBEDDING_API_URL is not set in environment variables")
                return

            results_text.insert(tk.END, "Searching for matching jobs...\n\n")
            root.update()

            matches, method = find_similar_jobs(cv_text, active_only=True)

            if not matches or method.startswith("Error"):
                results_text.insert(tk.END, "No matches found or search failed\n")
                return

            results_text.insert(tk.END, f"Found {len(matches)} potential matches:\n\n")
            
            for i, job in enumerate(matches):
                results_text.insert(tk.END, f"{i+1}. {job.get('Title', 'Unknown Position')}\n")
                results_text.insert(tk.END, f"   Company: {job.get('Company', 'N/A')}\n")
                results_text.insert(tk.END, f"   Location: {job.get('Area', 'N/A')}\n")
                results_text.insert(tk.END, f"   Posted: {job.get('posting_date', 'N/A')}\n")
                results_text.insert(tk.END, f"   Status: {job.get('Status', 'unknown')}\n")
                results_text.insert(tk.END, f"   Match score: {job.get('score', 0):.2f}\n")
                results_text.insert(tk.END, f"   URL: {job.get('url', '#')}\n\n")

            # Save results to JSON
            import json
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            json_filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                       "data", 
                                       f"job_suggestions_{timestamp}.json")
            
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "search_method": method,
                "total_matches": len(matches),
                "matches": matches
            }
            
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            results_text.insert(tk.END, f"\nResults saved to {json_filename}\n")
            showinfo("Success", "Job matching complete!")

        except Exception as e:
            showerror("Error", f"An error occurred: {str(e)}")

    # Create UI elements
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # File selection
    ttk.Label(frame, text="CV File:").grid(row=0, column=0, sticky=tk.W)
    file_entry = ttk.Entry(frame, width=60)
    file_entry.grid(row=0, column=1, padx=5)
    ttk.Button(frame, text="Browse", command=select_file).grid(row=0, column=2)

    # Process button
    ttk.Button(frame, text="Process CV", command=process_cv).grid(row=1, column=0, columnspan=3, pady=10)

    # Results area
    results_text = scrolledtext.ScrolledText(frame, width=80, height=30)
    results_text.grid(row=2, column=0, columnspan=3, pady=5)

    # Configure grid
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)

    return root

def main():
    root = create_ui()
    root.mainloop()

if __name__ == "__main__":
    main()