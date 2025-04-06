import sqlite3
import os   
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "job_listings.db")

def delete_embeddings_column():
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create a temporary table without the embeddings column
        cursor.execute('''
            CREATE TABLE jobs_temp AS 
            SELECT id, title, url, application_url, company_id, area_id, published_date, 
                   category_id, created_at, html_content
            FROM jobs;
        ''')

        # Drop the original table
        cursor.execute('DROP TABLE jobs;')

        # Rename the temporary table to the original name
        cursor.execute('ALTER TABLE jobs_temp RENAME TO jobs;')

        # Commit the changes
        conn.commit()
        print("Successfully removed embeddings column from jobs table")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    delete_embeddings_column()