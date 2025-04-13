import axios from 'axios';

// Environment variables (will need to be set in Vercel)
const EMBEDDING_API_URL = process.env.EMBEDDING_API_URL || '';
const CHROMA_HOST = process.env.CHROMA_HOST || '';
const CHROMA_PORT = parseInt(process.env.CHROMA_PORT || '8000');
const COLLECTION_NAME = process.env.CHROMA_COLLECTION || '';
const TOP_N_RESULTS = parseInt(process.env.TOP_N_RESULTS || '10');
const VERIFY_SSL = process.env.VERIFY_SSL === 'true';

export async function processCV(cvText) {
  console.log('Starting CV processing...');
  
  if (!EMBEDDING_API_URL) {
    throw new Error('EMBEDDING_API_URL environment variable is not set');
  }
  
  if (!CHROMA_HOST) {
    throw new Error('CHROMA_HOST environment variable is not set');
  }
  
  if (!COLLECTION_NAME) {
    throw new Error('CHROMA_COLLECTION environment variable is not set');
  }
  
  try {
    // Generate embedding for the CV
    console.log('Generating CV embedding...');
    const cvEmbedding = await generateCvEmbeddingRemote(cvText);
    
    if (!cvEmbedding) {
      throw new Error('Failed to generate CV embedding');
    }
    
    console.log('CV embedding generated successfully');
    
    // Find similar jobs
    console.log('Finding similar jobs...');
    const { matches, method } = await findSimilarJobs(cvEmbedding);
    
    console.log(`Found ${matches.length} matching jobs`);
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
    throw new Error(`Embedding generation failed: ${error.message}`);
  }
}

// Add this function at the beginning of the file, after the imports
async function safeJsonParse(response) {
  try {
    return response.data;
  } catch (error) {
    console.error('Error parsing JSON response:', error);
    console.error('Response data:', response.data);
    throw new Error(`Invalid JSON response: ${error.message}`);
  }
}

// Then modify the getRemoteEmbedding function
async function getRemoteEmbedding(texts) {
  try {
    if (!EMBEDDING_API_URL) {
      throw new Error("EMBEDDING_API_URL is not set");
    }
    
    console.log(`Calling remote embedding API at ${EMBEDDING_API_URL}`);
    
    const response = await axios.post(
      EMBEDDING_API_URL, 
      { texts }, 
      { 
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        validateStatus: false,
        timeout: 30000, // 30 second timeout
        httpsAgent: VERIFY_SSL ? undefined : new (require('https').Agent)({ rejectUnauthorized: false })
      }
    );
    
    if (response.status !== 200) {
      console.error(`API returned status ${response.status}`);
      console.error('Response data:', response.data);
      throw new Error(`API returned status ${response.status}: ${typeof response.data === 'string' ? response.data : JSON.stringify(response.data)}`);
    }
    
    // Safely parse the JSON response
    const data = await safeJsonParse(response);
    const embeddings = data.embeddings || [];
    
    if (embeddings.length === 0) {
      console.warn("Warning: Empty embedding response from API");
      return null;
    }
    
    return embeddings;
  } catch (error) {
    console.error('Error calling embedding API:', error);
    throw new Error(`Embedding API call failed: ${error.message}`);
  }
}

async function findSimilarJobs(cvEmbedding) {
  try {
    console.log(`Connecting to ChromaDB at ${CHROMA_HOST}:${CHROMA_PORT}`);
    
    // Try different API endpoint formats to find the correct one
    let chromaUrl = `http://${CHROMA_HOST}:${CHROMA_PORT}/api/v1/collections/${COLLECTION_NAME}/query`;
    console.log(`Attempting connection to: ${chromaUrl}`);
    
    const requestData = {
      query_embeddings: [cvEmbedding],
      n_results: TOP_N_RESULTS,
      include: ["metadatas", "distances", "documents"],
      where: { "Status": "active" }
    };
    
    console.log('Request payload:', JSON.stringify(requestData).substring(0, 200) + '...');
    
    let response;
    try {
      response = await axios({
        method: 'post',
        url: chromaUrl,
        data: requestData,
        headers: {
          "Accept": "application/json", 
          "Content-Type": "application/json"
        },
        timeout: 30000, // 30 second timeout
        validateStatus: false // Don't throw on error status codes
      });
      
      console.log(`ChromaDB response status: ${response.status}`);
      
      // If we get a 405, try the v2 API format
      if (response.status === 405) {
        console.log('Received 405 Method Not Allowed, trying v2 API format');
        chromaUrl = `http://${CHROMA_HOST}:${CHROMA_PORT}/api/v1/query`;
        
        const v2RequestData = {
          collection_name: COLLECTION_NAME,
          query_embeddings: [cvEmbedding],
          n_results: TOP_N_RESULTS,
          include: ["metadatas", "distances", "documents"],
          where: { "Status": "active" }
        };
        
        response = await axios({
          method: 'post',
          url: chromaUrl,
          data: v2RequestData,
          headers: {
            "Accept": "application/json", 
            "Content-Type": "application/json"
          },
          timeout: 30000,
          validateStatus: false
        });
        
        console.log(`ChromaDB v2 API response status: ${response.status}`);
      }
    } catch (connectionError) {
      console.error('Connection error:', connectionError);
      throw new Error(`Failed to connect to ChromaDB: ${connectionError.message}`);
    }
    
    if (response.status !== 200) {
      console.error(`ChromaDB returned status ${response.status}`);
      if (typeof response.data === 'string') {
        console.error('Response data:', response.data.substring(0, 500));
      } else {
        console.error('Response data:', JSON.stringify(response.data).substring(0, 500));
      }
      throw new Error(`ChromaDB returned status ${response.status}`);
    }
    
    const results = response.data;
    
    if (!results.metadatas || !results.metadatas[0] || results.metadatas[0].length === 0) {
      console.warn("No matching jobs found in ChromaDB");
      return { matches: [], method: "ChromaDB Vector Search (No matches)" };
    }
    
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
    throw new Error(`ChromaDB search failed: ${error.message}`);
  }
}