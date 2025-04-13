# CV Job Matcher

A web application that allows users to upload their CV and find matching job opportunities using vector similarity search.

## Features

- CV file upload (.txt, .pdf, .docx)
- Job matching using vector embeddings
- Display of matching jobs with scores
- Links to apply for jobs

## Environment Variables

The following environment variables need to be set in Vercel:

- `EMBEDDING_API_URL`: URL of the embedding API
- `CHROMA_HOST`: Hostname of the ChromaDB server
- `CHROMA_PORT`: Port of the ChromaDB server
- `CHROMA_COLLECTION`: Name of the ChromaDB collection
-