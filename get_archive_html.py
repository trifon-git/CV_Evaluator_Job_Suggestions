async def process_jobs_in_batches(batch_size=10):
    """Process jobs from database in batches."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Increase SQLite limits
        cursor.execute('PRAGMA max_page_count = 2147483646')
        cursor.execute('PRAGMA page_size = 32768')
        
        # Get count of remaining jobs
        cursor.execute('''
            SELECT COUNT(*) FROM jobs 
            WHERE html_content IS NULL 
            AND application_url IS NOT NULL
        ''')
        remaining_jobs = cursor.fetchone()[0]
        log_message(f"Found {remaining_jobs} unprocessed jobs")
        
        processed_count = 0
        while processed_count < remaining_jobs:
            # Get next batch of unprocessed jobs
            cursor.execute('''
                SELECT id, application_url 
                FROM jobs 
                WHERE html_content IS NULL 
                AND application_url IS NOT NULL
                ORDER BY id
                LIMIT ? OFFSET ?
            ''', (batch_size, processed_count))
            
            jobs = cursor.fetchall()
            if not jobs:
                break
                
            log_message(f"Processing jobs {processed_count+1} to {processed_count+len(jobs)} of {remaining_jobs}")
            
            tasks = [scrape_job_content(None, job_id, url) for job_id, url in jobs if url]
            if tasks:
                await asyncio.gather(*tasks)
            
            processed_count += len(jobs)
            await asyncio.sleep(1)
            
            # Periodically log progress
            if processed_count % 100 == 0:
                log_message(f"Progress: {processed_count}/{remaining_jobs} jobs processed")
            
        conn.close()
        log_message(f"Finished processing all {processed_count} jobs!")
        
    except Exception as e:
        log_message(f"Database error: {e}")