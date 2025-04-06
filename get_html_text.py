import os
import sqlite3
import nest_asyncio
import asyncio
import aiofiles
from pathlib import Path
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from asyncio import Semaphore
from datetime import datetime

# Apply nest_asyncio for compatibility with Async environments
nest_asyncio.apply()

# Increase concurrent tasks
SEMAPHORE = Semaphore(10)  # Increased from 3 to 10 concurrent tasks
LOG_FILE = "get_archive_html_process_log.txt"
DB_PATH = "job_listings.db"

def log_message(message):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {message}")
    with open("error_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"{timestamp} - {message}\n")


def clean_text_with_beautifulsoup(html_content):
    """Extract plain text from HTML using BeautifulSoup."""
    return BeautifulSoup(html_content, 'html.parser').get_text(separator=' ', strip=True)


async def extract_html_from_frames(page):
    """Extract raw HTML content from all frames on the page."""
    try:
        # Wait for body to be available
        await page.wait_for_selector('body', timeout=5000)
        
        # Get content from main frame first
        main_content = await page.evaluate("""() => document.body.innerHTML""")
        
        # Then try to get content from other frames
        frame_contents = []
        for frame in page.frames[1:]:  # Skip main frame
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
    async with SEMAPHORE:
        async with async_playwright() as p:
            try:
                browser = await p.firefox.launch(headless=True)
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                )
                page = await context.new_page()

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        log_message(f"[Job ID: {job_id}] Attempt {attempt + 1}: Accessing {url}")
                        
                        # More generous initial timeout
                        await page.goto(url, timeout=15000, wait_until='networkidle')
                        
                        # Try to get content
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
                        
                        # Longer delay between retries
                        await asyncio.sleep(3)
                
                return None
                
            except Exception as e:
                log_message(f"[Job ID: {job_id}] Critical error: {str(e)}")
                return None
            finally:
                try:
                    await browser.close()
                except:
                    pass

def init_database():
    """Initialize database with new html_content column."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if html_content column exists
    cursor.execute("PRAGMA table_info(jobs)")
    columns = [column[1] for column in cursor.fetchall()]
    
    # Add html_content column only if it doesn't exist
    if 'html_content' not in columns:
        cursor.execute('''
            ALTER TABLE jobs ADD COLUMN html_content TEXT;
        ''')
        conn.commit()
    
    return conn

async def scrape_job_content(conn, job_id, url):
    """Scrape and save content for a single job."""
    try:
        cleaned_text = await scrape_and_clean_text(url, job_id)
        if cleaned_text:
            # Create a new connection for each update
            update_conn = sqlite3.connect(DB_PATH)
            cursor = update_conn.cursor()
            try:
                cursor.execute('''
                    UPDATE jobs 
                    SET html_content = ? 
                    WHERE id = ?
                ''', (cleaned_text, job_id))
                update_conn.commit()
                log_message(f"Updated content for job ID: {job_id}")
            finally:
                update_conn.close()
        else:
            log_message(f"No content retrieved for job ID: {job_id}")
    except Exception as e:
        log_message(f"Error processing job ID {job_id}: {e}")

async def process_jobs_in_batches(batch_size=10):
    """Process jobs from database in batches."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get total count first
        cursor.execute('''
            SELECT COUNT(*) FROM jobs 
            WHERE html_content IS NULL 
            AND application_url IS NOT NULL
        ''')
        total_jobs = cursor.fetchone()[0]
        
        # Process in smaller chunks with offset
        offset = 0
        while offset < total_jobs:
            cursor.execute('''
                SELECT id, application_url 
                FROM jobs 
                WHERE html_content IS NULL 
                AND application_url IS NOT NULL
                ORDER BY id
                LIMIT ? OFFSET ?
            ''', (batch_size, offset))
            
            jobs = cursor.fetchall()
            if not jobs:
                break
                
            log_message(f"Processing jobs {offset+1} to {offset+len(jobs)} of {total_jobs}...")
            
            # Process one batch at a time
            tasks = [scrape_job_content(None, job_id, url) for job_id, url in jobs if url]
            if tasks:
                await asyncio.gather(*tasks)
            
            offset += batch_size
            await asyncio.sleep(1)  # Small delay between batches
            
        conn.close()
        log_message("All jobs processed successfully!")
        
    except Exception as e:
        log_message(f"Database error: {e}")

async def main():
    try:
        # Initialize database with new column
        conn = init_database()
        conn.close()
        
        # Process jobs in batches
        await process_jobs_in_batches()
        
    except Exception as e:
        log_message(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())