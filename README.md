# 🇩🇰 CV Job Matcher: Denmark

## Workflow / Pipeline Steps

The project operates through a sequence of steps, managed by different scripts:

1.  **Initial Data Population (Run Manually/Infrequently):**
    *   **`[STEP1]jobindex_scraper.py`**:
        *   Connects to MongoDB.
        *   Iterates through predefined Jobindex categories (`subid`).
        *   Scrapes *recent* job listings (controlled by `JOB_AGE_PARAM` in `.env`) for each category.
        *   Extracts basic metadata: Title, Jobindex URL, Application URL, Company, Area, Published Date.
        *   Assigns a unique `job_id` and `Category`.
        *   Inserts new, unique jobs into the MongoDB collection (skips duplicates based on Title, URL, Company index).
    *   **`[STEP2]get_html_text.py`**:
        *   Connects to MongoDB.
        *   Finds jobs in MongoDB that have an `Application_URL` but *lack* the `html_content` field.
        *   Uses Playwright (headless browser) to navigate to the `Application_URL` for each job (in batches, concurrently).
        *   Extracts the full text content from the page (handling frames, basic PDF/special URL detection).
        *   Cleans the extracted HTML into plain text using BeautifulSoup.
        *   Updates the corresponding job document in MongoDB with the extracted `html_content`.
    *   **`[STEP3]import_mongo_to_chroma.py`**:
        *   Connects to MongoDB and ChromaDB.
        *   Fetches jobs from MongoDB that have `html_content` but whose `_id` is *not* already present in the ChromaDB collection.
        *   For each new job (in batches):
            *   Calls the remote `EMBEDDING_API_URL` to generate a vector embedding for the `html_content`.
            *   Prepares metadata (Title, URLs, Company, Area, Category, Dates, Status - defaults to 'active' initially).
            *   Adds the embedding, metadata, and the `html_content` itself as a document to ChromaDB, using the MongoDB `_id` as the ChromaDB ID.

2.  **Keeping Data Fresh (Run Periodically, e.g., Daily):**
    *   **`[LOOP]jobindex_scraper_CHECK.py`**:
        *   Connects to MongoDB and ChromaDB.
        *   Scrapes *all* currently listed job URLs from Jobindex across all categories (does *not* filter by age like STEP 1). This generates a set of `active_job_urls`.
        *   **Updates MongoDB:**
            *   Sets `Status` to `"inactive"` for *all* jobs in the MongoDB collection.
            *   Sets `Status` to `"active"` for jobs whose `URL` is present in the `active_job_urls` set.
        *   **Updates ChromaDB:**
            *   Iterates through jobs in MongoDB (in batches).
            *   For each job's ID, fetches its existing metadata from ChromaDB.
            *   Updates the `Status` field in the ChromaDB metadata to match the status found in MongoDB. This ensures the `active_only` filter in `cv_match.py` works correctly.
    *   **`[LOOP]_backup_mongo.py`**:
        *   Connects to MongoDB.
        *   Creates a full backup of the main `MONGO_COLLECTION` into a new collection named like `backup_daily_YYYYMMDD_HHMMSS`.
        *   Rotates old backups based on retention settings (`DAILY_RETENTION`, `WEEKLY_RETENTION`, `MONTHLY_RETENTION`) defined in `.env`.

3.  **User Interaction (Running Continuously):**
    *   **`app.py`**:
        *   Starts the Streamlit web server.
        *   Provides a file uploader for users to submit their CV (PDF, DOCX, TXT, MD).
        *   Reads and extracts text content from the uploaded CV.
        *   Calls `cv_match.py:find_similar_jobs()`, passing the CV text.
    *   **`cv_match.py:find_similar_jobs()`**:
        *   Takes the CV text.
        *   Generates a vector embedding for the CV text using the remote `EMBEDDING_API_URL` (handles chunking for long CVs).
        *   Connects to ChromaDB.
        *   Queries the ChromaDB collection using the CV embedding.
        *   **Crucially, applies a `where` filter `{"Status": "active"}` by default** (can be overridden) to search only within active jobs.
        *   Retrieves the top N most similar job embeddings, their metadata, distances, and documents (content).
        *   Calculates a percentage similarity score based on the distance.
        *   Returns the list of matched jobs (including Title, Company, URL, Score, Status, etc.) back to `app.py`.
    *   **`app.py` (Continued):**
        *   Displays the matched jobs to the user in a formatted way, including links, scores, and status.

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.8+
    *   MongoDB server running and accessible.
    *   ChromaDB server running and accessible.
    *   Access to a remote text embedding API endpoint.
    *   Git (for cloning).

2.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

3.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Install Playwright Browsers:**
    Playwright needs browser binaries. Run this command to install default browsers (like Chromium):
    ```bash
    playwright install
    ```
    Or `playwright install chromium` if you only need Chromium (as used in `get_html_text.py`).

6.  **Configure Environment Variables:**
    

## Running the Pipeline & Application

1.  **Initial Data Load:**
    Run these scripts sequentially the first time, or whenever you need to refresh the entire base dataset:
    ```bash
    python "[STEP1]jobindex_scraper.py"
    python "[STEP2]get_html_text.py"
    python "[STEP3]import_mongo_to_chroma.py"
    ```
    *Note: Step 2 can take a significant amount of time depending on the number of jobs and network speed.*

2.  **Periodic Updates (Automation Recommended):**
    Schedule these scripts to run regularly (e.g., using `cron` on Linux/macOS or Task Scheduler on Windows):
    *   **Daily Status Check:**
        ```bash
        python "[LOOP]jobindex_scraper_CHECK.py"
        ```
    *   **Daily Backup:**
        ```bash
        python "[LOOP]_backup_mongo.py" # Add logic or separate script if you need weekly/monthly types too
        ```

3.  **Running the Web Application:**
    Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```
    Access the application through the URL provided by Streamlit (usually `http://localhost:8501`).


