from chromadb import HttpClient
from tabulate import tabulate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# === CONFIG ===
CHROMA_HOST = os.getenv('CHROMA_HOST')
CHROMA_PORT = int(os.getenv('CHROMA_PORT', '8000'))
COLLECTION_NAME = os.getenv('CHROMA_COLLECTION')

# === CONNECT TO CHROMADB ===
chroma_client = HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = chroma_client.get_collection(COLLECTION_NAME)

# Get all items including documents (html_content)
results = collection.get(
    include=['metadatas', 'documents'],
    limit=5  # Reduced to 5 since we're showing more content
)

# Format the data for display
table_data = []
for i, (id, metadata, content) in enumerate(zip(results['ids'], results['metadatas'], results['documents']), 1):
    # Truncate content to first 200 characters
    preview = content[:200] + '...' if content else 'N/A'
    
    table_data.append([
        i,
        metadata.get('title', 'N/A'),
        metadata.get('company', 'N/A'),
        metadata.get('area', 'N/A'),
        preview
    ])

# Print statistics
print(f"\n=== ChromaDB Collection Statistics ===")
print(f"Total entries: {len(results['ids'])}")

# Print table of entries
print("\n=== Sample Job Listings ===")
print(tabulate(
    table_data,
    headers=['#', 'Title', 'Company', 'Area', 'Description Preview'],
    tablefmt='grid',
    maxcolwidths=[5, 30, 20, 15, 60]  # Control column widths
))