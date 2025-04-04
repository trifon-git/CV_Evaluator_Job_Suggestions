# Job Scraper and CV Matcher

## the script is set up to retrieve only a small sample rather than scraping everything available

#### subids_to_sample = ["1", "8"]  # Only 2 categories
### max_jobs_per_category = 3  # Only 3 jobs per category


This application consists of two main components:
1. A job scraper that collects job listings from JobIndex and stores them in a SQLite database
2. A CV matcher that compares your CV against the job listings to find the best matches

## Installation

### Prerequisites
- Python 3.x
- Virtual environment (recommended)

### Setup

1. Clone or download this repository
2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Install Playwright browsers:
   ```
   playwright install
   ```

## How It Works

### Job Scraper (`scraper_sqlite.py`)

The scraper collects job listings from JobIndex.dk and stores them in a SQLite database with the following features:

- Uses Playwright for browser automation to handle JavaScript-rendered content
- Extracts job details including title, company, location, and full description
- Generates embeddings for job descriptions using a sentence transformer model
- Stores data in a SQLite database with vector search capabilities (sqlite-vss)
- Supports concurrent scraping with configurable parameters

To run the scraper:
```
python scraper_sqlite.py
```

### CV Matcher (`cv_match.py`)

The CV matcher compares your CV against the job listings in the database to find the best matches:

- Reads your CV from a Markdown file (`cv_text_example.md`)
- Generates embeddings for your CV using the same sentence transformer model
- Performs vector similarity search to find jobs that match your skills and experience
- Supports both vector search (VSS) and fallback to manual cosine similarity calculation
- Displays the top matching jobs with similarity scores

To run the CV matcher:
```
python cv_match.py
```

## Customization

- Edit `cv_text_example.md` to include your own CV information
- Adjust parameters in the configuration sections of both scripts:
  - `MODEL_NAME`: The sentence transformer model to use
  - `TOP_N_RESULTS`: Number of top matches to display
  - Various scraping parameters in the `Config` class

## Dependencies

The main dependencies include:
- requests, beautifulsoup4: For web scraping
- pandas: For data manipulation
- readability-lxml: For extracting main content from web pages
- playwright: For browser automation
- sentence-transformers: For generating text embeddings
- sqlite-vss: For vector similarity search in SQLite
- numpy: For numerical operations

## Notes

- The first run of the scraper will create the database and populate it with job listings
- The CV matcher requires the database to be populated with job listings first
- The application uses a multilingual sentence transformer model that works with both English and Danish text
