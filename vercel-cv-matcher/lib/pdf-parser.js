import pdfParse from 'pdf-parse/lib/pdf-parse';

export async function parsePdf(buffer) {
  try {
    // Use pdf-parse directly without relying on the test files
    const data = await pdfParse(buffer, {
      // Provide minimal options to avoid test file dependencies
      max: 0, // No page limit
    });
    
    return { 
      text: data.text || '',
      info: data.info,
      metadata: data.metadata
    };
  } catch (error) {
    console.error('Error parsing PDF:', error);
    throw new Error(`PDF parsing failed: ${error.message}`);
  }
}