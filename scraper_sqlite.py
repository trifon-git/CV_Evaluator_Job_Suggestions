import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import json
from urllib.parse import urljoin
import traceback
import sqlite3
import numpy as np
import os
import concurrent.futures
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import threading

# Dependencies
try:
    from readability import Document
    from playwright.sync_api import sync_playwright
    import sqlite_vss
    from sentence_transformers import SentenceTransformer
    DEPENDENCIES_LOADED = True
except ImportError as e:
    print(f"Warning: Missing dependency: {e}")
    print("Install with: pip install readability-lxml playwright sentence-transformers sqlite-vss")
    print("Also run: playwright install")
    DEPENDENCIES_LOADED = False

# Configuration
@dataclass
class Config:
    db_path: str = "job_listings.db"
    model_name: str = 'paraphrase-multilingual-mpnet-base-v2'
    request_timeout: int = 30
    min_desc_length: int = 100
    max_workers: int = 4
    delay_between_requests: float = 0.7
    delay_after_scraping: float = 1.2
    delay_between_categories: float = 2.5
    headers: Dict[str, str] = field(default_factory=lambda: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json, text/html, application/xhtml+xml, application/xml;q=0.9, */*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Referer': 'https://www.jobindex.dk/',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    })

# Category Data
MAIN_CATEGORY_IDS = { 
    "Information Technology": 1, 
    "Engineering and technology": 2, 
    "Management and staff": 3, 
    "Trade and service": 4, 
    "Industry and craft": 5, 
    "Sales and communication": 6, 
    "Teaching": 7, 
    "Office and finance": 8, 
    "Social and health": 9, 
    "Other positions": 10 
}

# Subcategories
SUBCATEGORY_DATA = {
    "93": ("Database", "Information Technology"), 
    "2": ("Financial and company systems", "Information Technology"), 
    "6": ("Internet and WWW", "Information Technology"), 
    "4": ("IT maintenance and support", "Information Technology"), 
    "3": ("IT Management", "Information Technology"), 
    "1": ("System development and programming", "Information Technology"), 
    "7": ("Tele- and datacommunication", "Information Technology"), 
    "8": ("Building and grounds technology", "Engineering and technology"),
    "9": ("Chemical engineering", "Engineering and technology"),
    "10": ("Electrical engineering", "Engineering and technology"),
    "11": ("Mechanical engineering", "Engineering and technology"),
    "12": ("Industrial design", "Engineering and technology"),
    "13": ("Process engineering", "Engineering and technology"),
    "14": ("Production planning", "Engineering and technology"),
    "15": ("Technical design", "Engineering and technology"),
    "16": ("Other technical areas", "Engineering and technology"),
    "17": ("Project management", "Management and staff"),
    "18": ("Middle management", "Management and staff"),
    "19": ("Executive management", "Management and staff"),
    "20": ("Board positions", "Management and staff"),
    "21": ("HR and personnel", "Management and staff"),
    "22": ("Training and education", "Management and staff"),
    "23": ("Purchasing", "Trade and service"),
    "24": ("Transportation and logistics", "Trade and service"),
    "25": ("Warehouse and distribution", "Trade and service"),
    "26": ("Catering and hospitality", "Trade and service"),
    "27": ("Retail", "Trade and service"),
    "28": ("Property and real estate", "Trade and service"),
    "29": ("Other service areas", "Trade and service"),
    "30": ("Agriculture and forestry", "Industry and craft"),
    "31": ("Construction and maintenance", "Industry and craft"),
    "32": ("Energy and utilities", "Industry and craft"),
    "33": ("Food production", "Industry and craft"),
    "34": ("Manual trades", "Industry and craft"),
    "35": ("Manufacturing", "Industry and craft"),
    "36": ("Advertising and marketing", "Sales and communication"),
    "37": ("PR and communication", "Sales and communication"),
    "38": ("Sales", "Sales and communication"),
    "39": ("Early childhood education", "Teaching"),
    "40": ("Elementary education", "Teaching"),
    "41": ("Secondary education", "Teaching"),
    "42": ("Higher education", "Teaching"),
    "43": ("Special education", "Teaching"),
    "44": ("Vocational training", "Teaching"),
    "45": ("Accounting", "Office and finance"),
    "46": ("Administrative support", "Office and finance"),
    "47": ("Banking and finance", "Office and finance"),
    "48": ("Customer service", "Office and finance"),
    "49": ("Insurance", "Office and finance"),
    "50": ("Legal", "Office and finance"),
    "51": ("Secretarial", "Office and finance"),
    "52": ("Dentistry", "Social and health"),
    "53": ("Healthcare", "Social and health"),
    "54": ("Medicine", "Social and health"),
    "55": ("Nursing", "Social and health"),
    "56": ("Psychology and counseling", "Social and health"),
    "57": ("Social work", "Social and health"),
    "58": ("Architecture", "Other positions"),
    "59": ("Arts and culture", "Other positions"),
    "60": ("Environmental", "Other positions"),
    "61": ("Journalism and media", "Other positions"),
    "62": ("Research and development", "Other positions"),
    "63": ("Student jobs", "Other positions"),
    "64": ("Volunteer work", "Other positions"),
}

# Thread-local storage for database connections
local_storage = threading.local()

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
        print(f"Loading embedding model: {self.model_name}...")
        try:
            self.model = SentenceTransformer(self.model_name)
            self.vector_dimension = self.model.get_sentence_embedding_dimension()
            self.max_seq_length = self.model.max_seq_length
            self.tokenizer = self.model.tokenizer
            
            self.chunk_size = self.max_seq_length
            self.chunk_overlap = max(50, self.chunk_size // 4)
            
            print(f"Model loaded. Vector dimension: {self.vector_dimension}, Max sequence length: {self.max_seq_length}")
            print(f"Using chunk size: {self.chunk_size}, overlap: {self.chunk_overlap}")
            
            return True
        except Exception as e:
            print(f"[CRITICAL] Failed to load model '{self.model_name}': {e}")
            traceback.print_exc()
            return False
            
    def get_embedding(self, text):
        if not text or not isinstance(text, str):
            print("Cannot generate embedding: empty or invalid text")
            return None
            
        try:
            tokens = self.tokenizer.encode(text)
            total_tokens = len(tokens)
            
            if total_tokens <= self.max_seq_length:
                embedding = self.model.encode(text, normalize_embeddings=True)
                return embedding

            print(f"Text exceeds model limit. Chunking ({total_tokens} tokens)...")
            chunk_embeddings = []
            start_index = 0
            step = self.chunk_size - self.chunk_overlap
            
            while start_index < total_tokens:
                end_index = start_index + self.max_seq_length
                chunk_token_ids = tokens[start_index:end_index]
                chunk_text = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
                
                if not chunk_text.strip():
                    start_index += step
                    continue
                    
                chunk_embedding = self.model.encode(chunk_text, normalize_embeddings=True)
                chunk_embeddings.append(chunk_embedding)
                start_index += step

            if not chunk_embeddings:
                print("No valid chunks generated")
                return None
                
            average_embedding = np.mean(chunk_embeddings, axis=0)
            norm = np.linalg.norm(average_embedding)
            
            if norm == 0:
                print("Warning: Norm of average embedding is zero")
                return average_embedding
                
            return average_embedding / norm
            
        except Exception as e:
            print(f"Error during embedding generation: {e}")
            return None

class DatabaseManager:
    def __init__(self, db_path, vector_dimension=None):
        self.db_path = db_path
        self.vector_dimension = vector_dimension
        self.vss_enabled = False
        self.conn_lock = threading.RLock()
        
    def get_connection(self):
        if not hasattr(local_storage, 'db_conn'):
            with self.conn_lock:
                try:
                    conn = sqlite3.connect(self.db_path, timeout=30)
                    conn.row_factory = sqlite3.Row
                    
                    try:
                        if hasattr(conn, 'enable_load_extension'):
                            conn.enable_load_extension(True)
                            sqlite_vss.load(conn)
                            print(f"Thread {threading.get_ident()}: VSS extension loaded")
                        else:
                            print(f"Thread {threading.get_ident()}: SQLite build doesn't support extensions")
                    except Exception as e:
                        print(f"Thread {threading.get_ident()}: VSS extension loading failed: {e}")
                    
                    local_storage.db_conn = conn
                    print(f"Thread {threading.get_ident()}: Created new database connection")
                except Exception as e:
                    print(f"Thread {threading.get_ident()}: Failed to create database connection: {e}")
                    raise
        
        return local_storage.db_conn
    
    def initialize(self):
        print(f"Initializing database at: {self.db_path}")
        
        try:
            with self.conn_lock:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                try:
                    if hasattr(conn, 'enable_load_extension'):
                        conn.enable_load_extension(True)
                        sqlite_vss.load(conn)
                        print("SQLite VSS extension loaded successfully")
                        self.vss_enabled = True
                    else:
                        print("[WARNING] This SQLite build doesn't support loading extensions")
                        print("Continuing without vector search capability")
                        self.vss_enabled = False
                except sqlite3.Error as e:
                    print(f"[WARNING] Failed to load sqlite-vss extension: {e}")
                    print("Continuing without vector search capability")
                    self.vss_enabled = False
                    
                self._create_tables(cursor)
                
                if self.vss_enabled and self.vector_dimension:
                    try:
                        cursor.execute(f'''CREATE VIRTUAL TABLE IF NOT EXISTS vss_jobs 
                                        USING vss0(embedding({self.vector_dimension}));''')
                        print(f"VSS table created for dimension {self.vector_dimension}")
                    except sqlite3.Error as e:
                        print(f"[WARNING] Failed to create VSS table: {e}")
                        print("Vector search will be disabled")
                        self.vss_enabled = False
                else:
                    print("VSS extension not available - vector search will be disabled")
                
                conn.commit()
                print("Database initialization complete")
                return True
            
        except sqlite3.Error as e:
            print(f"[CRITICAL] Database initialization failed: {e}")
            traceback.print_exc()
            return False
    
    def _create_tables(self, cursor):
        cursor.execute('''CREATE TABLE IF NOT EXISTS categories ( 
            Main_Category_ID INTEGER PRIMARY KEY, 
            Main_Category_Name TEXT NOT NULL UNIQUE 
        )''')
        
        cursor.execute("SELECT COUNT(*) FROM categories")
        if cursor.fetchone()[0] == 0:
            print("Populating main categories table...")
            for name, cat_id in MAIN_CATEGORY_IDS.items(): 
                cursor.execute("INSERT OR IGNORE INTO categories (Main_Category_ID, Main_Category_Name) VALUES (?, ?)", 
                              (cat_id, name))
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS companies ( 
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            Company TEXT NOT NULL UNIQUE 
        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS areas ( 
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            Area TEXT NOT NULL UNIQUE 
        )''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS jobs ( 
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            Title TEXT NOT NULL, 
            Company_ID INTEGER, 
            Area_ID INTEGER, 
            Published_Date TEXT, 
            Specific_Category_Name TEXT, 
            Main_Category_ID INTEGER, 
            SubID TEXT, 
            JobIndexURL TEXT UNIQUE NOT NULL, 
            Full_ListingURL TEXT, 
            Full_Description TEXT, 
            embedding BLOB, 
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
            FOREIGN KEY (Company_ID) REFERENCES companies (id), 
            FOREIGN KEY (Area_ID) REFERENCES areas (id), 
            FOREIGN KEY (Main_Category_ID) REFERENCES categories (Main_Category_ID) 
        )''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_jobindexurl ON jobs(JobIndexURL)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_main_cat_id ON jobs(Main_Category_ID)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_published ON jobs(Published_Date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_companies_name ON companies(Company)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_areas_name ON areas(Area)')
    
    def insert_job(self, job_data, embedding_vector):
        with self.conn_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            inserted = False
            job_rowid = None
            
            try:
                company_name = job_data.get('Company', 'N/A')
                cursor.execute('INSERT OR IGNORE INTO companies (Company) VALUES (?)', (company_name,))
                cursor.execute('SELECT id FROM companies WHERE Company = ?', (company_name,))
                company_result = cursor.fetchone()
                company_id = company_result['id'] if company_result else None
                
                area_name = job_data.get('Area')
                area_id = None
                if area_name: 
                    cursor.execute('INSERT OR IGNORE INTO areas (Area) VALUES (?)', (area_name,))
                    cursor.execute('SELECT id FROM areas WHERE Area = ?', (area_name,))
                    area_result = cursor.fetchone()
                    area_id = area_result['id'] if area_result else None
                
                embedding_blob = self._vector_to_blob(embedding_vector)
                
                cursor.execute('''
                    INSERT OR IGNORE INTO jobs (
                        Title, Company_ID, Area_ID, Published_Date, 
                        Specific_Category_Name, Main_Category_ID, SubID, 
                        JobIndexURL, Full_ListingURL, Full_Description, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    job_data.get('Title', 'N/A'),
                    company_id,
                    area_id,
                    job_data.get('Published_Date'),
                    job_data.get('Specific_Category_Name'),
                    job_data.get('Main_Category_ID'),
                    job_data.get('SubID'),
                    job_data.get('JobIndexURL'),
                    job_data.get('Full_ListingURL'),
                    job_data.get('Full_Description'),
                    embedding_blob
                ))
                
                if cursor.rowcount > 0: 
                    inserted = True
                    job_rowid = cursor.lastrowid
                else: 
                    cursor.execute("SELECT rowid FROM jobs WHERE JobIndexURL = ?", (job_data.get('JobIndexURL'),))
                    existing_job = cursor.fetchone()
                    job_rowid = existing_job['rowid'] if existing_job else None

                if self.vss_enabled and job_rowid and embedding_blob:
                    try:
                        cursor.execute("INSERT OR IGNORE INTO vss_jobs (rowid, embedding) VALUES (?, ?);", 
                                      (job_rowid, embedding_blob))
                    except Exception as e: 
                        print(f"Failed to insert into VSS table: {e}")
                
                conn.commit()
                
                return inserted
                
            except Exception as e: 
                print(f"Database insertion failed: {e}")
                conn.rollback()
                return False
            
    def _vector_to_blob(self, vector):
        if vector is None:
            return None
        return np.array(vector, dtype=np.float32).tobytes()
        
    def _blob_to_vector(self, blob):
        if blob is None:
            return None
        return np.frombuffer(blob, dtype=np.float32)
        
    def commit(self):
        with self.conn_lock:
            if hasattr(local_storage, 'db_conn'):
                local_storage.db_conn.commit()
            
    def close(self):
        with self.conn_lock:
            if hasattr(local_storage, 'db_conn'):
                local_storage.db_conn.close()
                delattr(local_storage, 'db_conn')
            
    def close_all(self):
        self.close()
            
    def get_sample_data(self, limit=15):
        with self.conn_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    SELECT j.Title, c.Company, a.Area, j.Published_Date, j.Specific_Category_Name, j.JobIndexURL,
                           CASE WHEN j.Full_Description IS NOT NULL 
                                THEN SUBSTR(j.Full_Description, 1, 60) || '...' 
                                ELSE NULL 
                           END AS Description_Snippet,
                           CASE WHEN j.embedding IS NOT NULL 
                                THEN 'Yes (' || LENGTH(j.embedding) || ' bytes)' 
                                ELSE 'No' 
                           END AS Has_Embedding
                    FROM jobs j 
                    LEFT JOIN companies c ON j.Company_ID = c.id 
                    LEFT JOIN areas a ON j.Area_ID = a.id
                    ORDER BY j.scraped_at DESC 
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                if rows:
                    return [dict(row) for row in rows]
                return []
                
            except Exception as e:
                print(f"Error fetching sample data: {e}")
                return None

class JobScraper:
    def __init__(self, config, model_manager, db_manager):
        self.config = config
        self.model = model_manager
        self.db = db_manager
        self.session = requests.Session()
        self.session.headers.update(config.headers)
        self.session_lock = threading.RLock()
        
    def get_category_map(self):
        final_mapping = {}
        processed_subids = set()
        
        for subid, (specific_name, main_category_name) in SUBCATEGORY_DATA.items():
            if subid in processed_subids:
                continue
                
            if main_category_name in MAIN_CATEGORY_IDS:
                main_cat_id = MAIN_CATEGORY_IDS[main_category_name]
                final_mapping[subid] = {"id": main_cat_id, "category": specific_name}
                processed_subids.add(subid)
            else:
                print(f"Warning: Main category '{main_category_name}' not found for subid {subid}")
                
        print(f"Processed {len(final_mapping)} category mappings")
        return final_mapping
        
    def get_final_job_url(self, jobindex_vis_job_url):
        with self.session_lock:
            try:
                response_initial = self.session.get(jobindex_vis_job_url, timeout=self.config.request_timeout)
                response_initial.raise_for_status()
                
                soup = BeautifulSoup(response_initial.text, 'html.parser')
                tracking_link = soup.find('a', class_='seejobdesktop') or soup.find('a', class_='eemobiljob')
                
                if not tracking_link or not tracking_link.get('href'):
                    return None
                    
                tracking_url = urljoin(jobindex_vis_job_url, tracking_link['href'])
                
                response_final = self.session.get(
                    tracking_url, 
                    timeout=self.config.request_timeout, 
                    allow_redirects=True
                )
                response_final.raise_for_status()
                
                return response_final.url
                
            except requests.exceptions.RequestException as e:
                print(f"Request error resolving URL: {e}")
                return None
            except Exception as e:
                print(f"Unexpected error resolving URL: {e}")
                return None
    
    def scrape_description(self, url):
        if not url:
            return None
            
        # Method 1: Requests + Readability
        with self.session_lock:
            try:
                response = self.session.get(url, timeout=self.config.request_timeout)
                response.raise_for_status()
                
                content_type = response.headers.get('Content-Type', '').lower()
                if 'html' not in content_type:
                    raise ValueError("Not HTML content")
                    
                doc = Document(response.text)
                summary_html = doc.summary()
                soup = BeautifulSoup(summary_html, 'html.parser')
                description = soup.get_text(separator='\n', strip=True)
                
                if description and len(description) >= self.config.min_desc_length:
                    return description
                    
            except Exception as e:
                print(f"Method 1 (Requests+Readability) failed: {e}")
        
        # Method 2: Playwright + Readability
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page(user_agent=self.config.headers['User-Agent'])
                
                page.goto(url, timeout=60000, wait_until='domcontentloaded')
                page.wait_for_timeout(1500)
                
                html_content = page.content()
                browser.close()
                
                doc = Document(html_content)
                summary_html = doc.summary()
                soup = BeautifulSoup(summary_html, 'html.parser')
                description = soup.get_text(separator='\n', strip=True)
                
                if description and len(description) >= self.config.min_desc_length:
                    return description
                    
        except Exception as e:
            print(f"Method 2 (Playwright+Readability) failed: {e}")
            
        return None
        
    def process_job(self, job_ad, subid, category_info):
        try:
            thread_id = threading.get_ident()
            print(f"Thread {thread_id} processing job")
            
            share_element = job_ad.find('div', class_='jobad-element-menu-share')
            jobindex_vis_url = share_element.get('data-share-url') if share_element else None
            
            if not jobindex_vis_url:
                return False
                
            title = share_element.get('data-share-title', 'N/A') if share_element else 'N/A'
            company_div = job_ad.find('div', class_='jix-toolbar-top__company')
            company_name = company_div.find('a').text.strip() if company_div and company_div.find('a') else 'N/A'
            location_span = job_ad.find('span', class_='jix_robotjob--area')
            area = location_span.get_text(strip=True) if location_span else None
            published_tag = job_ad.find('time')
            published_date = published_tag.get('datetime') if published_tag else None
            
            full_listing_url = self.get_final_job_url(jobindex_vis_url)
            time.sleep(self.config.delay_between_requests)
            
            full_description = self.scrape_description(full_listing_url)
            time.sleep(self.config.delay_after_scraping)
            
            embedding = None
            if full_description:
                embedding = self.model.get_embedding(full_description)
            
            job_data = {
                "Title": title,
                "Company": company_name,
                "Area": area,
                "Published_Date": published_date,
                "Specific_Category_Name": category_info.get("category", "Unknown"),
                "Main_Category_ID": category_info.get("id", 0),
                "SubID": subid,
                "JobIndexURL": jobindex_vis_url,
                "Full_ListingURL": full_listing_url,
                "Full_Description": full_description,
            }
            
            success = self.db.insert_job(job_data, embedding)
            print(f"Thread {thread_id} completed job processing, success: {success}")
            return success
            
        except Exception as e:
            print(f"Error in thread {threading.get_ident()}: {e}")
            traceback.print_exc()
            return False
        
    def process_category(self, subid, category_info, keyword="", max_jobs=3):
        specific_category = category_info.get("category", "Unknown")
        print(f"\n--- Processing Category: '{specific_category}' (SubID: {subid}) ---")
        
        base_url = f"https://www.jobindex.dk/jobsoegning.json?subid={subid}"
        processed_count = 0
        params = {"q": keyword, "page": 1}
        
        try:
            with self.session_lock:
                response = self.session.get(base_url, params=params, timeout=self.config.request_timeout)
                response.raise_for_status()
                
                if 'application/json' not in response.headers.get('Content-Type', ''):
                    print("Unexpected content type")
                    return 0
                    
                data = response.json()
                html_content = data.get('result_list_box_html', '')
            
            if not html_content:
                print("No HTML content")
                return 0
                
            soup = BeautifulSoup(html_content, 'html.parser')
            job_ads = soup.find_all('div', class_='jobsearch-result')
            
            if not job_ads:
                print("No job listings found")
                return 0
                
            print(f"Found {len(job_ads)} listings. Processing up to {max_jobs}...")
            jobs_to_process = job_ads[:max_jobs]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_job = {
                    executor.submit(self.process_job, job, subid, category_info): job 
                    for job in jobs_to_process
                }
                
                for future in concurrent.futures.as_completed(future_to_job):
                    try:
                        if future.result():
                            processed_count += 1
                    except Exception as e:
                        print(f"Job processing error: {e}")
                        traceback.print_exc()
            
        except Exception as e:
            print(f"Error processing category: {e}")
            traceback.print_exc()
            
        print(f"--- Finished '{specific_category}', added {processed_count} new jobs to DB")
        return processed_count

def main():
    print("Starting JobIndex Scraper...")
    start_time = time.time()
    
    if not DEPENDENCIES_LOADED:
        print("Critical dependencies missing. Please install them first")
        return
    
    config = Config()
    
    model_manager = EmbeddingModel(config.model_name)
    if not model_manager.load():
        print("Model loading failed. Exiting")
        return
        
    db_manager = DatabaseManager(config.db_path, model_manager.vector_dimension)
    if not db_manager.initialize():
        print("Database initialization failed. Exiting")
        return
    
    try:
        scraper = JobScraper(config, model_manager, db_manager)
        
        category_map = scraper.get_category_map()
        if not category_map:
            print("No categories available. Exiting")
            return
            
        subids_to_sample = ["1", "8"]
        max_jobs_per_category = 3
        total_new_jobs = 0
        
        print(f"\nSampling {len(subids_to_sample)} categories, up to {max_jobs_per_category} jobs each")
        
        for subid in subids_to_sample:
            if subid in category_map:
                category_info = category_map[subid]
                new_jobs = scraper.process_category(
                    subid, 
                    category_info,
                    keyword="", 
                    max_jobs=max_jobs_per_category
                )
                total_new_jobs += new_jobs
                time.sleep(config.delay_between_categories)
            else:
                print(f"Warning: SubID {subid} not found in category map. Skipping")
                
        print(f"\n--- Processing complete. Added {total_new_jobs} new jobs ---")
        
        print("\nFetching sample data from database...")
        sample_data = db_manager.get_sample_data(15)
        
        if sample_data:
            db_df = pd.DataFrame(sample_data)
            print("\nLast ~15 Jobs Added/Checked in DB:")
            with pd.option_context('display.max_colwidth', 50, 'display.max_rows', 25):
                print(db_df)
        else:
            print("No data found or could not retrieve sample")
            
    except Exception as e:
        print(f"Critical error in main execution: {e}")
        traceback.print_exc()
    finally:
        if 'db_manager' in locals():
            db_manager.close_all()
            print("Database connections closed")

if __name__ == "__main__":
    main()