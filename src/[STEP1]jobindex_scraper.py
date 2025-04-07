import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import sqlite3
from datetime import datetime
import json

category_job_id = 1
category_name = "active"

# Database setup
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "job_listings.db")

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create categories table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY,
            subid TEXT NOT NULL,
            name TEXT NOT NULL,
            url_path TEXT NOT NULL
        )
    ''')
    
    # Create companies table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
    ''')
    
    # Create areas table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS areas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
    ''')
    
    # Create jobs table with foreign keys - removed AUTOINCREMENT
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            url TEXT NOT NULL UNIQUE,
            application_url TEXT,
            company_id INTEGER,
            area_id INTEGER,
            published_date TEXT NOT NULL,
            category_id INTEGER,
            created_at TIMESTAMP,
            FOREIGN KEY (category_id) REFERENCES categories (id),
            FOREIGN KEY (company_id) REFERENCES companies (id),
            FOREIGN KEY (area_id) REFERENCES areas (id)
        )
    ''')
    
    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_url ON jobs(url)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_category ON jobs(category_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_published ON jobs(published_date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_companies_name ON companies(name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_areas_name ON areas(name)')
    
    conn.commit()
    return conn

def insert_categories(conn, subid_mapping):
    cursor = conn.cursor()
    for subid, info in subid_mapping.items():
        cursor.execute('''
            INSERT OR REPLACE INTO categories (id, subid, name, url_path)
            VALUES (?, ?, ?, ?)
        ''', (info['id'], subid, info['category'], f"/jobsoegning/{info['category']}"))
    conn.commit()

def insert_jobs(conn, jobs_data, category_id):
    cursor = conn.cursor()
    
    # Get the current maximum ID from the jobs table
    cursor.execute('SELECT MAX(id) FROM jobs')
    result = cursor.fetchone()
    last_id = result[0] if result[0] is not None else 0
    
    for job in jobs_data:
        try:
            # Get current timestamp for created_at
            current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Insert or get company
            cursor.execute('INSERT OR IGNORE INTO companies (name) VALUES (?)', (job['Company'],))
            cursor.execute('SELECT id FROM companies WHERE name = ?', (job['Company'],))
            company_id = cursor.fetchone()[0]
            
            # Insert or get area
            cursor.execute('INSERT OR IGNORE INTO areas (name) VALUES (?)', (job['Area'],))
            cursor.execute('SELECT id FROM areas WHERE name = ?', (job['Area'],))
            area_id = cursor.fetchone()[0]
            
            # Check if the job URL already exists
            cursor.execute('SELECT id FROM jobs WHERE url = ?', (job['URL'],))
            existing_job = cursor.fetchone()
            
            if existing_job:
                # Update existing job
                cursor.execute('''
                    UPDATE jobs 
                    SET title = ?, application_url = ?, company_id = ?, area_id = ?, 
                        published_date = ?, category_id = ?
                    WHERE url = ?
                ''', (
                    job['Title'],
                    job['Application_URL'],
                    company_id,
                    area_id,
                    job['Published'],
                    category_id,
                    job['URL']
                ))
            else:
                # Increment the ID for new job
                last_id += 1
                
                # Insert new job with manually assigned ID and explicit created_at timestamp
                cursor.execute('''
                    INSERT INTO jobs 
                    (id, title, url, application_url, company_id, area_id, published_date, category_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    last_id,
                    job['Title'],
                    job['URL'],
                    job['Application_URL'],
                    company_id,
                    area_id,
                    job['Published'],
                    category_id,
                    current_timestamp
                ))
        except sqlite3.IntegrityError:
            # Skip duplicates
            continue
    conn.commit()

JOB_AGE_PARAM = "jobage=1"

def fetch_job_listings(keyword, page_limit=1, subid="1"):
    base_url = f"https://www.jobindex.dk/jobsoegning.json?subid={subid}&{JOB_AGE_PARAM}"
    job_listings = []

    for page in range(1, page_limit + 1):
        params = {
            "q": keyword,
            "page": page
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            html_content = data.get('result_list_box_html', '')
            soup = BeautifulSoup(html_content, 'html.parser')

            # Find all job ads
            job_ads = soup.find_all('div', class_='jobsearch-result')
            
            # Stop if no job postings are found
            if not job_ads:
                print(f"No job postings found on page {page}. Stopping early.")
                break

            for job_ad in job_ads:
                # Extract title and URLs
                share_div = job_ad.find('div', class_='jobad-element-menu-share')
                title = share_div['data-share-title']
                jobindex_url = share_div['data-share-url']
                
                # Extract direct application URL - look specifically for the job title link
                application_link = job_ad.find('h4').find('a', href=True)
                application_url = application_link['href'] if application_link else None

                job_location = job_ad.find('span', class_='jix_robotjob--area')

                # Extract company name with better error handling
                company_div = job_ad.find('div', class_='jix-toolbar-top__company')
                if company_div and company_div.find('a'):
                    company_name = company_div.find('a').text.strip()
                else:
                    company_name = 'Unknown'

                # Extract the publication date
                published_tag = job_ad.find('time')
                published_date = published_tag['datetime'] if published_tag else 'Unknown'

                # Append all required data to the list
                job_listings.append({
                    "Title": title,
                    "URL": jobindex_url,
                    "Application_URL": application_url,
                    "Category_Job_ID": category_job_id,
                    "Category_Job": category_name,
                    "Company": company_name,
                    "Area": job_location.get_text(strip=True) if job_location else 'Unknown',
                    "Published": published_date
                })

    return job_listings

def update_json_if_new(output_file, new_data, update_threshold=5):
    try:
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error loading JSON file: {output_file}. Starting with an empty dataset.")
        existing_data = []

    # Combine data and remove duplicates
    combined_data = pd.DataFrame(existing_data + new_data).drop_duplicates(subset=["Title", "URL"]).to_dict(orient="records")

    # Identify new rows
    new_rows = [row for row in new_data if row not in existing_data]

    # Update only if new rows exceed the threshold
    if len(new_rows) >= update_threshold:
        try:
            with open(output_file, 'w') as f:
                json.dump(combined_data, f, indent=4)
            print(f"JSON updated with {len(new_rows)} new rows.")
        except Exception as e:
            print(f"Error writing to JSON file: {output_file}. Error: {e}")
    else:
        print(f"Only {len(new_rows)} new rows found. Waiting for {update_threshold - len(new_rows)} more.")

def scrape_jobs(conn, keyword, category_id, page_limit=1, subid="1"):
    print(f"Scraping data for keyword: {keyword} with subid: {subid}...")
    try:
        new_data = fetch_job_listings(keyword, page_limit, subid)
        if new_data:
            insert_jobs(conn, new_data, category_id)
            print(f"Data saved to database for category {category_id}.")
        else:
            print("No data found.")
    except Exception as e:
        print(f"Error scraping data: {e}")

def main():
    try:
        # Initialize database
        conn = init_database()
        
        # Get all available subids
        SUBID_MAPPING = get_all_subids()
        if not SUBID_MAPPING:
            print("Failed to fetch subids. Exiting...")
            return

        # Insert categories into database
        insert_categories(conn, SUBID_MAPPING)

        keyword = ""  # Empty string to get all jobs
        page_limit = 1000

        # Iterate through each subid
        for subid, info in SUBID_MAPPING.items():
            print(f"\nScraping jobs for subid {subid} ({info['category']})...")
            scrape_jobs(conn, keyword, info['id'], page_limit, subid)

        print("\nAll categories completed.")
        conn.close()
    except Exception as e:
        print(f"Critical error in the scraping pipeline: {e}")

def get_all_subids():
    # Load the complete subid data
    subid_data = {
        "1": {"1": "/jobsoegning/it/systemudvikling"},
        "2": {"1": "/jobsoegning/it/virksomhedssystemer"},
        "3": {"1": "/jobsoegning/it/itledelse", "3": "/jobsoegning/it/itledelse"},
        "4": {"1": "/jobsoegning/it/itdrift"},
        "6": {"1": "/jobsoegning/it/internet"},
        "7": {"1": "/jobsoegning/it/telekom"},
        "8": {"2": "/jobsoegning/ingenioer/byggeteknik"},
        "10": {"2": "/jobsoegning/ingenioer/medicinal"},
        "11": {"2": "/jobsoegning/ingenioer/elektronik"},
        "12": {"3": "/jobsoegning/ledelse/personale"},
        "13": {"3": "/jobsoegning/ledelse/topledelse"},
        "14": {"3": "/jobsoegning/ledelse/leder"},
        "15": {"10": "/jobsoegning/oevrige/oevrige"},
        "16": {"9": "/jobsoegning/social/laege"},
        "17": {"9": "/jobsoegning/social/sygeplejerske"},
        "18": {"8": "/jobsoegning/kontor/kontor"},
        "21": {"8": "/jobsoegning/kontor/sekretaer"},
        "22": {"10": "/jobsoegning/kontor/kontorelev", "8": "/jobsoegning/kontor/kontorelev"},
        "23": {"4": "/jobsoegning/handel/boernepasning"},
        "24": {"5": "/jobsoegning/industri/byggeri"},
        "25": {"5": "/jobsoegning/industri/landbrug"},
        "26": {"3": "/jobsoegning/ledelse/freelancekonsulent"},
        "27": {"7": "/jobsoegning/undervisning/paedagog"},
        "28": {"7": "/jobsoegning/undervisning/laerer"},
        "33": {"8": "/jobsoegning/kontor/oekonomi"},
        "35": {"8": "/jobsoegning/kontor/finans"},
        "36": {"5": "/jobsoegning/industri/naeringsmiddel"},
        "37": {"7": "/jobsoegning/undervisning/bibliotek"},
        "38": {"8": "/jobsoegning/kontor/offentlig", "9": "/jobsoegning/kontor/offentlig"},
        "40": {"5": "/jobsoegning/industri/toemrer"},
        "41": {"9": "/jobsoegning/social/terapi"},
        "44": {"5": "/jobsoegning/industri/industri"},
        "45": {"7": "/jobsoegning/undervisning/forskning"},
        "47": {"9": "/jobsoegning/social/pleje"},
        "49": {"6": "/jobsoegning/salg/kommunikation"},
        "51": {"9": "/jobsoegning/social/tandlaege"},
        "52": {"8": "/jobsoegning/kontor/jura"},
        "53": {"8": "/jobsoegning/kontor/indkoeb"},
        "54": {"8": "/jobsoegning/kontor/logistik"},
        "55": {"6": "/jobsoegning/salg/marketing"},
        "56": {"4": "/jobsoegning/handel/frisoer"},
        "57": {"6": "/jobsoegning/salg/telemarketing"},
        "58": {"6": "/jobsoegning/salg/salg"},
        "60": {"6": "/jobsoegning/salg/ejendomsmaegler", "8": "/jobsoegning/salg/ejendomsmaegler"},
        "61": {"3": "/jobsoegning/ledelse/projektledelse"},
        "63": {"9": "/jobsoegning/social/laegesekretaer"},
        "65": {"6": "/jobsoegning/salg/kultur"},
        "67": {"4": "/jobsoegning/handel/hotel"},
        "70": {"4": "/jobsoegning/handel/detailhandel"},
        "71": {"4": "/jobsoegning/handel/service"},
        "73": {"4": "/jobsoegning/handel/rengoering"},
        "74": {"5": "/jobsoegning/industri/lager"},
        "75": {"3": "/jobsoegning/ledelse/salgschef", "6": "/jobsoegning/ledelse/salgschef"},
        "77": {"9": "/jobsoegning/social/socialraadgivning"},
        "78": {"10": "/jobsoegning/oevrige/elev"},
        "79": {"3": "/jobsoegning/ledelse/institutions", "7": "/jobsoegning/ledelse/institutions"},
        "80": {"5": "/jobsoegning/industri/elektriker"},
        "81": {"3": "/jobsoegning/ledelse/oekonomichef", "8": "/jobsoegning/ledelse/oekonomichef"},
        "83": {"5": "/jobsoegning/industri/transport"},
        "84": {"10": "/jobsoegning/oevrige/student"},
        "85": {"2": "/jobsoegning/ingenioer/maskiningenioer"},
        "89": {"6": "/jobsoegning/salg/grafisk"},
        "90": {"5": "/jobsoegning/industri/jern"},
        "91": {"9": "/jobsoegning/social/psykologi"},
        "92": {"5": "/jobsoegning/industri/tekstil"},
        "93": {"1": "/jobsoegning/it/database"},
        "94": {"2": "/jobsoegning/ingenioer/kemi"},
        "95": {"5": "/jobsoegning/industri/mekanik"},
        "96": {"5": "/jobsoegning/industri/blik"},
        "97": {"5": "/jobsoegning/industri/maling"},
        "98": {"4": "/jobsoegning/handel/ejendomsservice", "8": "/jobsoegning/handel/ejendomsservice"},
        "99": {"4": "/jobsoegning/handel/bud"},
        "100": {"9": "/jobsoegning/social/teknisksundhed"},
        "103": {"7": "/jobsoegning/undervisning/voksenuddannelse"},
        "104": {"5": "/jobsoegning/industri/traeindustri"},
        "106": {"8": "/jobsoegning/kontor/oversaettelse"},
        "110": {"6": "/jobsoegning/salg/design"},
        "112": {"4": "/jobsoegning/handel/sikkerhed"},
        "113": {"10": "/jobsoegning/oevrige/frivilligt"},
        "120": {"6": "/jobsoegning/salg/franchise"},
        "121": {"2": "/jobsoegning/ingenioer/teknikledelse", "3": "/jobsoegning/ingenioer/teknikledelse"},
        "122": {"2": "/jobsoegning/ingenioer/produktionsteknik"},
        "123": {"10": "/jobsoegning/oevrige/studiepraktik"},
        "124": {"3": "/jobsoegning/ledelse/detailledelse", "4": "/jobsoegning/ledelse/detailledelse"},
        "125": {"3": "/jobsoegning/ledelse/virksomhedsudvikling"},
        "126": {"8": "/jobsoegning/kontor/akademisk"},
        "127": {"10": "/jobsoegning/oevrige/forsvar"},
        "85": {"2": "/jobsoegning/ingenioer/maskiningenioer"},
        "89": {"6": "/jobsoegning/salg/grafisk"},
        "90": {"5": "/jobsoegning/industri/jern"},
        "91": {"9": "/jobsoegning/social/psykologi"},
        "92": {"5": "/jobsoegning/industri/tekstil"},
        "93": {"1": "/jobsoegning/it/database"},
        "94": {"2": "/jobsoegning/ingenioer/kemi"},
        "95": {"5": "/jobsoegning/industri/mekanik"},
        "96": {"5": "/jobsoegning/industri/blik"},
        "97": {"5": "/jobsoegning/industri/maling"},
        "98": {"4": "/jobsoegning/handel/ejendomsservice", "8": "/jobsoegning/handel/ejendomsservice"},
        "99": {"4": "/jobsoegning/handel/bud"}
    }
    
    try:
        subid_mapping = {}
        for subid, categories in subid_data.items():
            cat_id = next(iter(categories))
            url = categories[cat_id]
            category = url.split('/')[-1]
            
            subid_mapping[subid] = {
                "id": int(cat_id),
                "category": category
            }
        
        print(f"Found {len(subid_mapping)} categories to scrape")
        return subid_mapping
    except Exception as e:
        print(f"Error processing subids: {e}")
        return {}

if __name__ == "__main__":
    main()