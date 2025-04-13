const axios = require('axios');

const CHROMA_HOST = process.env.CHROMA_HOST || 'your-chroma-host';
const CHROMA_PORT = process.env.CHROMA_PORT || '8000';
const COLLECTION_NAME = process.env.CHROMA_COLLECTION || 'your-collection';

async function testChromaConnection() {
  console.log(`Testing ChromaDB connection to ${CHROMA_HOST}:${CHROMA_PORT}`);
  
  try {
    // Test GET collections endpoint
    const collectionsUrl = `http://${CHROMA_HOST}:${CHROMA_PORT}/api/v1/collections`;
    console.log(`Checking collections at: ${collectionsUrl}`);
    
    const collectionsResponse = await axios.get(collectionsUrl);
    console.log('Collections response:', collectionsResponse.status);
    console.log('Collections data:', JSON.stringify(collectionsResponse.data).substring(0, 500));
    
    // Test specific collection
    const collectionUrl = `http://${CHROMA_HOST}:${CHROMA_PORT}/api/v1/collections/${COLLECTION_NAME}`;
    console.log(`Checking collection at: ${collectionUrl}`);
    
    const collectionResponse = await axios.get(collectionUrl);
    console.log('Collection response:', collectionResponse.status);
    console.log('Collection data:', JSON.stringify(collectionResponse.data).substring(0, 500));
    
    // Test both query formats
    const v1QueryUrl = `http://${CHROMA_HOST}:${CHROMA_PORT}/api/v1/collections/${COLLECTION_NAME}/query`;
    const v2QueryUrl = `http://${CHROMA_HOST}:${CHROMA_PORT}/api/v1/query`;
    
    // Create a simple test embedding (all zeros)
    const testEmbedding = new Array(1536).fill(0);
    
    // Test v1 query format
    try {
      console.log(`Testing v1 query at: ${v1QueryUrl}`);
      const v1Response = await axios.post(v1QueryUrl, {
        query_embeddings: [testEmbedding],
        n_results: 1
      }, {
        headers: { "Content-Type": "application/json" },
        validateStatus: false
      });
      console.log('V1 query response:', v1Response.status);
    } catch (v1Error) {
      console.error('V1 query error:', v1Error.message);
    }
    
    // Test v2 query format
    try {
      console.log(`Testing v2 query at: ${v2QueryUrl}`);
      const v2Response = await axios.post(v2QueryUrl, {
        collection_name: COLLECTION_NAME,
        query_embeddings: [testEmbedding],
        n_results: 1
      }, {
        headers: { "Content-Type": "application/json" },
        validateStatus: false
      });
      console.log('V2 query response:', v2Response.status);
    } catch (v2Error) {
      console.error('V2 query error:', v2Error.message);
    }
    
  } catch (error) {
    console.error('Error testing ChromaDB connection:', error.message);
  }
}

testChromaConnection();