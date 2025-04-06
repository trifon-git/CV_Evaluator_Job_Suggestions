import sqlite3
import os

def view_jobs_with_content():
    conn = sqlite3.connect(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "job_listings.db"))
    cursor = conn.cursor()

    # Count total jobs
    cursor.execute('SELECT COUNT(*) FROM jobs')
    total_jobs = cursor.fetchone()[0]

    # Count jobs with HTML content
    cursor.execute('SELECT COUNT(*) FROM jobs WHERE html_content IS NOT NULL')
    jobs_with_content = cursor.fetchone()[0]

    # Count jobs without HTML content
    cursor.execute('SELECT COUNT(*) FROM jobs WHERE html_content IS NULL AND application_url IS NOT NULL')
    jobs_without_content = cursor.fetchone()[0]

    print("\nJob Statistics:")
    print("-" * 80)
    print(f"Total jobs in database: {total_jobs}")
    print(f"Jobs with HTML content: {jobs_with_content}")
    print(f"Jobs without HTML content (with URL): {jobs_without_content}")
    print(f"Success rate: {(jobs_with_content/total_jobs)*100:.2f}%")
    print("-" * 80)

    conn.close()

if __name__ == "__main__":
    view_jobs_with_content()