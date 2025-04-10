import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from pymongo import MongoClient
from tqdm import tqdm
from collections import defaultdict
from dotenv import load_dotenv
import chromadb
from chromadb import HttpClient

# Load environment variables
load_dotenv()

# === CONFIG ===
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('MONGO_DB_NAME')
COLLECTION_NAME = 'jobs'  # Collection to check against
CHROMA_HOST = os.getenv('CHROMA_HOST')
CHROMA_PORT = int(os.getenv('CHROMA_PORT'))
CHROMA_COLLECTION = os.getenv('CHROMA_COLLECTION')

# Add these variables after the existing CONFIG section
DATE_FORMAT = "%Y%m%d"
TIME_FORMAT = "%H%M%S"

def init_mongodb():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    edit_collection = db[COLLECTION_NAME]
    return client, edit_collection

def init_chromadb():
    try:
        chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collection = chroma_client.get_collection(CHROMA_COLLECTION)
        print(f"✅ Connected to ChromaDB collection: {CHROMA_COLLECTION}")
        return chroma_client, collection
    except Exception as e:
        print(f"❌ Error connecting to ChromaDB: {e}")
        return None, None

def update_job_status(edit_collection, active_job_urls):
    """Update job status in the collection based on active jobs"""
    # Mark all jobs as inactive first
    result = edit_collection.update_many(
        {}, 
        {"$set": {"Status": "inactive"}}
    )
    print(f"✅ Marked {result.modified_count} jobs as inactive")
    
    # Mark active jobs
    result = edit_collection.update_many(
        {"URL": {"$in": list(active_job_urls)}},
        {"$set": {"Status": "active"}}
    )
    print(f"✅ Marked {result.modified_count} jobs as active")
    
    # Get statistics
    total_jobs = edit_collection.count_documents({})
    active_jobs = edit_collection.count_documents({"Status": "active"})
    inactive_jobs = edit_collection.count_documents({"Status": "inactive"})
    
    print(f"\n=== JOB STATUS SUMMARY ===")
    print(f"Total jobs in database: {total_jobs}")
    print(f"Active jobs: {active_jobs} ({active_jobs/total_jobs*100:.2f}%)")
    print(f"Inactive jobs: {inactive_jobs} ({inactive_jobs/total_jobs*100:.2f}%)")
    
    return active_jobs

def update_chromadb_status(chroma_collection, mongo_collection):
    """Update ChromaDB with status information from MongoDB"""
    print("\nUpdating ChromaDB with job status information...")
    
    # Get all jobs from MongoDB with their status
    jobs = list(mongo_collection.find({}, {
        "_id": 1, 
        "job_id": 1, 
        "Title": 1, 
        "URL": 1, 
        "Status": 1
    }))
    
    # Update ChromaDB in batches
    batch_size = 100
    total_updated = 0
    
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i+batch_size]
        ids = [str(job["_id"]) for job in batch]
        metadatas = []
        
        for job in batch:
            # Get existing metadata from ChromaDB
            try:
                result = chroma_collection.get(ids=[str(job["_id"])], include=["metadatas"])
                if result and result["metadatas"]:
                    metadata = result["metadatas"][0]
                    # Add or update Status field
                    metadata["Status"] = job.get("Status", "inactive")
                    metadatas.append(metadata)
                else:
                    # If no existing metadata, create minimal metadata
                    metadatas.append({
                        "Title": job.get("Title", "Unknown"),
                        "URL": job.get("URL", ""),
                        "Status": job.get("Status", "inactive")
                    })
            except Exception as e:
                print(f"Error getting metadata for job {job['_id']}: {e}")
                metadatas.append({
                    "Title": job.get("Title", "Unknown"),
                    "URL": job.get("URL", ""),
                    "Status": job.get("Status", "inactive")
                })
        
        # Update metadata in ChromaDB
        try:
            chroma_collection.update(
                ids=ids,
                metadatas=metadatas
            )
            total_updated += len(batch)
            print(f"Updated {total_updated}/{len(jobs)} jobs in ChromaDB")
        except Exception as e:
            print(f"Error updating ChromaDB: {e}")
    
    print(f"✅ ChromaDB update complete. Updated {total_updated} jobs.")

def fetch_job_listings(keyword, page_limit=1, subid="1"):
    base_url = os.getenv('JOBINDEX_BASE_URL')
    
    base_url = f"{base_url}?subid={subid}"
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
                
                # Extract direct application URL
                application_link = job_ad.find('h4').find('a', href=True)
                application_url = application_link['href'] if application_link else None

                job_location = job_ad.find('span', class_='jix_robotjob--area')

                # Extract company name
                company_div = job_ad.find('div', class_='jix-toolbar-top__company')
                company_name = company_div.find('a').text.strip() if company_div and company_div.find('a') else 'Unknown'

                # Extract publication date
                published_tag = job_ad.find('time')
                # Convert published date to string format if it's not already
                published_date = published_tag['datetime'] if published_tag else 'Unknown'
                if published_date != 'Unknown':
                    try:
                        # Parse and format the date to ensure consistent string format
                        parsed_date = datetime.fromisoformat(published_date)
                        published_date = parsed_date.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        published_date = 'Unknown'

                job_listings.append({
                    "Title": title,
                    "URL": jobindex_url,
                    "Application_URL": application_url,
                    "Company": company_name,
                    "Area": job_location.get_text(strip=True) if job_location else 'Unknown',
                    "Published": published_date
                })

    return job_listings

def scrape_jobs(keyword, category, page_limit=1, subid="1", active_job_urls=None):
    print(f"Scraping data for category: {category} with subid: {subid}...")
    try:
        new_data = fetch_job_listings(keyword, page_limit, subid)
        if new_data:
            # Add all URLs to active jobs set
            for job in new_data:
                active_job_urls.add(job["URL"])
            print(f"✅ Found {len(new_data)} active jobs in this category")
        else:
            print("No data found.")
    except Exception as e:
        print(f"Error scraping data: {e}")

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

def main():
    try:
        # Initialize MongoDB connection
        mongo_client, edit_collection = init_mongodb()
        
        # Initialize ChromaDB connection
        chroma_client, chroma_collection = init_chromadb()
        if not chroma_collection:
            print("❌ Failed to connect to ChromaDB. Continuing with MongoDB only.")
        
        # Create a set to store active job URLs
        active_job_urls = set()
        
        # Get all available subids
        SUBID_MAPPING = get_all_subids()
        if not SUBID_MAPPING:
            print("Failed to fetch subids. Exiting...")
            return

        keyword = ""  # Empty string to get all jobs
        page_limit = 1000  # Check up to 1000 pages per category

        # Progress bar for categories
        categories = list(SUBID_MAPPING.items())
        for subid, info in tqdm(categories, desc="Processing categories"):
            print(f"\nProcessing category: {info['category']} (subid: {subid})")
            scrape_jobs(keyword, info['category'], page_limit, subid, active_job_urls)

        print("\n✅ All categories completed")
        print(f"Total active jobs found: {len(active_job_urls)}")
        
        # Update job status in MongoDB
        print("\nUpdating job status in MongoDB collection...")
        active_jobs_count = update_job_status(edit_collection, active_job_urls)
        
        # Update ChromaDB with status information
        if chroma_collection:
            update_chromadb_status(chroma_collection, edit_collection)
            print(f"\n✅ ChromaDB has been updated with status information.")
            print(f"[STEP4]cv_match.py can now filter for only active jobs ({active_jobs_count} jobs).")
        
        mongo_client.close()
        
    except Exception as e:
        print(f"❌ Critical error in the scraping pipeline: {e}")

if __name__ == "__main__":
    main()