import sqlite3
from datetime import datetime
import os

# Database setup
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "job_listings.db")

def log_message(message):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {message}")

def check_and_remove_duplicates(conn=None):
    """
    Check for and remove duplicate jobs, keeping only one instance.
    Jobs are considered duplicates if they have the same title, area_id, company_id, and category_id.
    """
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
    
    cursor = conn.cursor()
    
    try:
        # Find duplicates based on title and IDs
        log_message("Checking for duplicates based on title, area, company, and category...")
        cursor.execute('''
            SELECT title, area_id, company_id, category_id, COUNT(*) as count
            FROM jobs
            GROUP BY title, area_id, company_id, category_id
            HAVING COUNT(*) > 1
        ''')
        
        duplicates = cursor.fetchall()
        
        if not duplicates:
            log_message("No duplicate jobs found")
            return
            
        log_message(f"Found {len(duplicates)} sets of duplicates")
        
        total_deleted = 0
        
        for dup in duplicates:
            title, area_id, company_id, category_id, count = dup
            log_message(f"Processing duplicates for title: '{title}' (Area: {area_id}, Company: {company_id}, Category: {category_id})")
            
            # Get all IDs for these duplicates, ordered by ID
            cursor.execute('''
                SELECT id FROM jobs
                WHERE title = ? 
                AND area_id = ? 
                AND company_id = ? 
                AND category_id = ?
                ORDER BY id
            ''', (title, area_id, company_id, category_id))
            
            ids = [row[0] for row in cursor.fetchall()]
            
            # Keep the first ID and delete the rest
            keep_id = ids[0]
            delete_ids = ids[1:]
            
            if delete_ids:
                placeholders = ','.join('?' for _ in delete_ids)
                cursor.execute(f'''
                    DELETE FROM jobs
                    WHERE id IN ({placeholders})
                ''', delete_ids)
                
                conn.commit()
                log_message(f"Kept ID {keep_id}, deleted {len(delete_ids)} duplicates")
                total_deleted += len(delete_ids)
        
        log_message(f"\nDuplicate removal completed. Total deleted: {total_deleted}")
        
    except sqlite3.Error as e:
        log_message(f"Database error: {e}")
        conn.rollback()
    
    # Only close the connection if we created it in this function
    if conn is None:
        conn.close()

def check_database_stats():
    """Print statistics about the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Count total jobs
        cursor.execute("SELECT COUNT(*) FROM jobs")
        total_jobs = cursor.fetchone()[0]
        log_message(f"Total jobs in database: {total_jobs}")
        
        # Count jobs by category
        cursor.execute('''
            SELECT c.name, COUNT(j.id) 
            FROM jobs j
            JOIN categories c ON j.category_id = c.id
            GROUP BY c.name
            ORDER BY COUNT(j.id) DESC
        ''')
        
        category_counts = cursor.fetchall()
        log_message("\nJobs by category:")
        for category, count in category_counts:
            log_message(f"  {category}: {count}")
        
        # Count companies
        cursor.execute("SELECT COUNT(*) FROM companies")
        company_count = cursor.fetchone()[0]
        log_message(f"\nTotal companies: {company_count}")
        
        # Count areas
        cursor.execute("SELECT COUNT(*) FROM areas")
        area_count = cursor.fetchone()[0]
        log_message(f"Total areas: {area_count}")
        
    except sqlite3.Error as e:
        log_message(f"Database error: {e}")
    
    conn.close()

if __name__ == "__main__":
    log_message("Starting duplicate check process...")
    
    # Connect to the database
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Check database statistics before cleanup
        log_message("\n=== Database statistics before cleanup ===")
        check_database_stats()
        
        # Check and remove duplicates
        log_message("\n=== Starting duplicate removal ===")
        check_and_remove_duplicates(conn)
        
        # Check database statistics after cleanup
        log_message("\n=== Database statistics after cleanup ===")
        check_database_stats()
        
        conn.close()
        log_message("\nDuplicate check process completed successfully")
        
    except Exception as e:
        log_message(f"Error during duplicate check process: {e}")