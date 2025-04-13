import axios from 'axios';

// Environment variables (will need to be set in Vercel)
const EMBEDDING_API_URL = process.env.EMBEDDING_API_URL || '';
const CHROMA_HOST = process.env.CHROMA_HOST || '';
const CHROMA_PORT = parseInt(process.env.CHROMA_PORT || '8000');
const COLLECTION_NAME = process.env.CHROMA_COLLECTION || '';
const TOP_N_RESULTS = parseInt(process.env.TOP_N_RESULTS || '10');
const VERIFY_SSL = process.env.VERIFY_SSL === 'true';

export async function processCV(cvText) {
  try {
    // Generate embedding for the CV
    const cvEmbedding = await generateCvEmbeddingRemote(cvText);
    
    if (!cvEmbedding) {
      throw new Error('Failed to generate CV embedding');
    }
    
    // Find similar jobs
    const { matches, method } = await findSimilarJobs(cvEmbedding);
    
    return { matches, method };
  } catch (error) {
    console.error('Error in CV processing:', error);
    throw error;
  }
}

async function generateCvEmbeddingRemote(cvText) {
  try {
    // Split text into chunks
    const chunkSize = 1000;  // characters per chunk
    const overlap = 200;     // character overlap between chunks

    const chunks = [];
    let start = 0;
    while (start < cvText.length) {
      const end = start + chunkSize;
      const chunk = cvText.substring(start, end);
      if (chunk.trim()) {
        chunks.push(chunk);
      }
      start = end - overlap;
    }

    console.log(`Split CV into ${chunks.length} chunks for remote embedding`);

    const chunkEmbeddings = [];
    for (let i = 0; i < chunks.length; i++) {
      console.log(`Processing chunk ${i+1}/${chunks.length}`);
      const embeddingResponse = await getRemoteEmbedding([chunks[i]]);
      if (embeddingResponse && embeddingResponse.length > 0) {
        chunkEmbeddings.push(embeddingResponse[0]);
      } else {
        console.warn("⚠️ Skipped a chunk due to empty embedding response");
      }
    }

    if (chunkEmbeddings.length === 0) {
      console.error("Error: No valid chunks were embedded");
      return null;
    }

    // Average the embeddings
    const dimensions = chunkEmbeddings[0].length;
    const avgEmbedding = new Array(dimensions).fill(0);
    
    for (const embedding of chunkEmbeddings) {
      for (let i = 0; i < dimensions; i++) {
        avgEmbedding[i] += embedding[i] / chunkEmbeddings.length;
      }
    }
    
    // Normalize
    const norm = Math.sqrt(avgEmbedding.reduce((sum, val) => sum + val * val, 0));
    if (norm > 0) {
      for (let i = 0; i < dimensions; i++) {
        avgEmbedding[i] /= norm;
      }
    }
    
    return avgEmbedding;
  } catch (error) {
    console.error('Error generating remote embedding:', error);
    return null;
  }
}

async function getRemoteEmbedding(texts) {
  try {
    if (!EMBEDDING_API_URL) {
      console.error("Error: EMBEDDING_API_URL is not set");
      return null;
    }
    
    console.log(`Calling remote embedding API at ${EMBEDDING_API_URL}`);
    const response = await axios.post(EMBEDDING_API_URL, { texts }, {
      verify: VERIFY_SSL
    });
    const embeddings = response.data.embeddings || [];
    
    if (embeddings.length === 0) {
      console.warn("Warning: Empty embedding response from API");
      return null;
    }
    
    return embeddings;
  } catch (error) {
    console.error('Error calling embedding API:', error);
    return null;
  }
}

async function findSimilarJobs(cvEmbedding) {
  try {
    // Connect to ChromaDB
    const response = await axios.post(`http://${CHROMA_HOST}:${CHROMA_PORT}/api/v1/collections/${COLLECTION_NAME}/query`, {
      query_embeddings: [cvEmbedding],
      n_results: TOP_N_RESULTS,
      include: ["metadatas", "distances", "documents"],
      where: { "Status": "active" }
    }, {
      headers: {
        "accept": "application/json", 
        "Content-Type": "application/json"
      }
    });
    
    const results = response.data;
    const matches = [];
    
    for (let i = 0; i < results.metadatas[0].length; i++) {
      const metadata = results.metadatas[0][i];
      const distance = results.distances[0][i];
      const content = results.documents[0][i];
      
      // Using exponential decay for more intuitive scoring
      const similarityScore = Math.exp(-distance) * 100;  // Will give scores between 0-100
      
      matches.push({
        score: similarityScore,
        type: "ChromaDB similarity",
        Title: metadata.Title || 'Unknown Position',
        Company: metadata.Company || 'N/A',
        Area: metadata.Area || 'N/A',
        url: metadata.Application_URL || '#',
        posting_date: metadata.Published_Date || 'N/A',
        content: content,
        Status: metadata.Status || 'unknown'
      });
    }
    
    return { matches, method: "ChromaDB Vector Search" };
  } catch (error) {
    console.error('Error during ChromaDB search:', error);
    throw new Error('ChromaDB search failed');
  }
}