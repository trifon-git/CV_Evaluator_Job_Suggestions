// Create a new file for custom PDF parsing
import { PDFExtract } from 'pdf.js-extract';

export async function parsePdf(buffer) {
  try {
    const pdfExtract = new PDFExtract();
    const options = {};
    
    const data = await pdfExtract.extractBuffer(buffer, options);
    
    // Combine all page content
    let text = '';
    if (data && data.pages) {
      for (const page of data.pages) {
        if (page.content) {
          for (const item of page.content) {
            text += item.str + ' ';
          }
          text += '\n\n';
        }
      }
    }
    
    return { text: text.trim() };
  } catch (error) {
    console.error('Error parsing PDF:', error);
    throw new Error(`PDF parsing failed: ${error.message}`);
  }
}