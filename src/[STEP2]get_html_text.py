import os
import nest_asyncio
import asyncio
import aiofiles
from pathlib import Path
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from asyncio import Semaphore
from datetime import datetime
from pymongo import MongoClient
from tqdm import tqdm
from dotenv import load_dotenv

# Apply nest_asyncio for compatibility
nest_asyncio.apply()

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('MONGO_DB_NAME')
COLLECTION_NAME = os.getenv('MONGO_COLLECTION')
MAX_CONCURRENT = int(os.getenv('MAX_CONCURRENT'))  
SEMAPHORE = Semaphore(MAX_CONCURRENT)
PAGE_TIMEOUT = int(os.getenv('PAGE_TIMEOUT'))  
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))  

def log_message(message):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {message}")

def init_mongodb():
    """Initialize MongoDB connection."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    return client, collection

def clean_text_with_beautifulsoup(html_content):
    """Extract plain text from HTML using BeautifulSoup."""
    return BeautifulSoup(html_content, 'html.parser').get_text(separator=' ', strip=True)

async def extract_html_from_frames(page):
    """Extract raw HTML content from all frames on the page."""
    try:
        # Reduce timeout for faster failure detection
        await page.wait_for_selector('body', timeout=3000)  # Reduced from 5000ms
        main_content = await page.evaluate("""() => document.body.innerHTML""")
        
        # Only process the first 3 frames to save time
        frame_contents = []
        for frame in page.frames[1:4]:  # Limit to first 3 frames
            try:
                content = await frame.evaluate("""() => document.body.innerHTML""")
                frame_contents.append(content)
            except:
                continue
        return main_content + " " + " ".join(frame_contents)
    except Exception as e:
        log_message(f"Error extracting HTML from frames: {e}")
        return None

async def scrape_and_clean_text(url, job_id):
    """Scrape visible text from a webpage, clean it, and return it."""
    # Skip URLs containing 'youngcrm'
    if 'youngcrm' in url.lower():
        log_message(f"[Job ID: {job_id}] Skipping youngcrm URL: {url}")
        return "Skipped youngcrm URL"
    
    # More comprehensive PDF detection
    if url.lower().endswith('.pdf') or 'pdf' in url.lower():
        log_message(f"[Job ID: {job_id}] Likely PDF file: {url}")
        return f"PDF document available at: {url}"
        
    async with SEMAPHORE:
        async with async_playwright() as p:
            try:
                # Use chromium instead of firefox for faster loading
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    viewport={'width': 1280, 'height': 800},  # Smaller viewport for faster rendering
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                )
                page = await context.new_page()

                # Optimize page loading
                await page.route('**/*.{png,jpg,jpeg,gif,svg,css,woff,woff2,ttf}', lambda route: route.abort())  # Block unnecessary resources
                
                max_retries = 2  # Reduced from 3 for faster processing
                for attempt in range(max_retries):
                    try:
                        log_message(f"[Job ID: {job_id}] Attempt {attempt + 1}: Accessing {url}")
                        await page.goto(url, timeout=PAGE_TIMEOUT, wait_until='domcontentloaded')  # Changed from networkidle to domcontentloaded
                        html_content = await extract_html_from_frames(page)
                        if html_content:
                            return clean_text_with_beautifulsoup(html_content)
                        log_message(f"[Job ID: {job_id}] No content found on attempt {attempt + 1}")
                    except Exception as e:
                        log_message(f"[Job ID: {job_id}] Attempt {attempt + 1} failed: {str(e)}")
                        await page.close()
                        page = await context.new_page()
                        if attempt == max_retries - 1:
                            log_message(f"[Job ID: {job_id}] All attempts failed for {url}")
                            return None
                        await asyncio.sleep(1)  # Reduced from 3 seconds
                return None
            except Exception as e:
                log_message(f"[Job ID: {job_id}] Critical error: {str(e)}")
                return None
            finally:
                try:
                    await browser.close()
                except:
                    pass

async def process_jobs_in_batches(batch_size=None):
    """Process jobs from MongoDB in batches."""
    if batch_size is None:
        batch_size = BATCH_SIZE  # Use environment variable
        
    try:
        client, collection = init_mongodb()
        
        # Track failed job IDs to avoid repeated processing
        failed_job_ids = set()
        
        # Get total count of jobs to process
        total_jobs = collection.count_documents({
            'Application_URL': {'$ne': None},
            'html_content': {'$exists': False}
        })
        
        log_message(f"Found {total_jobs} jobs to process")
        processed_count = 0

        # Create a progress bar for the entire process
        progress_bar = tqdm(total=total_jobs, desc="Processing jobs", unit="job")

        while True:
            # Query for jobs without html_content, sorted by most recent first
            query = {
                'Application_URL': {'$ne': None},
                'html_content': {'$exists': False},
                '_id': {'$nin': list(failed_job_ids)}  # Skip previously failed jobs
            }

            # Get batch of jobs, sorted by creation date in descending order (newest first)
            jobs = list(collection.find(query).sort("Created_At", -1).limit(batch_size))
            
            if not jobs:
                log_message("No more jobs to process. Finished!")
                break

            log_message(f"Processing batch of {len(jobs)} jobs ({processed_count + len(jobs)}/{total_jobs})")
            
            # Process jobs concurrently
            tasks = [scrape_and_clean_text(job['Application_URL'], str(job['_id'])) for job in jobs]
            results = await asyncio.gather(*tasks)
            
            # Update database with results
            successful_updates = 0
            for job, cleaned_text in zip(jobs, results):
                if cleaned_text:
                    collection.update_one(
                        {'_id': job['_id']},
                        {'$set': {'html_content': cleaned_text}}
                    )
                    successful_updates += 1
                else:
                    # Mark as failed to avoid retrying
                    failed_job_ids.add(job['_id'])
                    log_message(f"Marking job {job['_id']} as failed to avoid retrying")
                processed_count += 1
                # Update progress bar
                progress_bar.update(1)
            
            log_message(f"Updated content for {successful_updates}/{len(jobs)} jobs in this batch")
            await asyncio.sleep(0.5)  # Reduced from 1 second

        # Close the progress bar
        progress_bar.close()
        log_message(f"Completed processing {processed_count} jobs. Successfully updated {processed_count} jobs.")

    except Exception as e:
        log_message(f"Database error: {e}")
    finally:
        client.close()

# Remove save_progress and load_progress functions

async def main():
    try:
        await process_jobs_in_batches()
    except Exception as e:
        log_message(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())