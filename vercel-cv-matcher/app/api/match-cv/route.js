import { NextResponse } from 'next/server';
import { processCV } from '../../../lib/cv-processor';
import { parsePdf } from '../../../lib/pdf-parser';

export async function POST(request) {
  try {
    const formData = await request.formData();
    const file = formData.get('cv');
    
    if (!file) {
      return NextResponse.json(
        { error: 'No CV file uploaded' },
        { status: 400 }
      );
    }

    // Read the file content
    const buffer = Buffer.from(await file.arrayBuffer());
    let cvText = '';
    
    try {
      // Check file type and extract text accordingly
      const fileName = file.name.toLowerCase();
      
      if (fileName.endsWith('.pdf')) {
        console.log('Processing PDF file');
        try {
          const pdfData = await parsePdf(buffer);
          cvText = pdfData.text;
          console.log('PDF parsing successful');
        } catch (pdfError) {
          console.error('PDF parsing error:', pdfError);
          return NextResponse.json(
            { error: `Failed to parse PDF: ${pdfError.message}` },
            { status: 500 }
          );
        }
      } else if (fileName.endsWith('.docx')) {
        // For now, we'll return an error for docx files
        return NextResponse.json(
          { error: 'DOCX files are not supported yet. Please upload a PDF or TXT file.' },
          { status: 400 }
        );
      } else {
        // Assume it's a text file
        console.log('Processing text file');
        cvText = buffer.toString('utf-8');
      }
      
      console.log(`CV text loaded (${cvText.length} characters)`);
      
      if (!cvText || cvText.trim().length === 0) {
        return NextResponse.json(
          { error: 'Could not extract text from the uploaded file' },
          { status: 400 }
        );
      }
    } catch (error) {
      console.error('Error reading file content:', error);
      return NextResponse.json(
        { error: 'Failed to read CV file content: ' + error.message },
        { status: 500 }
      );
    }
    
    // Process the CV using our matching algorithm
    try {
      console.log('Starting CV processing...');
      const results = await processCV(cvText);
      console.log('CV processing completed successfully');
      return NextResponse.json({ results });
    } catch (error) {
      console.error('Error in CV processing:', error);
      // Return a more detailed error message
      return NextResponse.json(
        { error: `Failed to process CV: ${error.message || 'Unknown error'}` },
        { status: 500 }
      );
    }
  } catch (error) {
    console.error('Error processing CV request:', error);
    return NextResponse.json(
      { error: 'Failed to process CV request: ' + (error.message || 'Unknown error') },
      { status: 500 }
    );
  }
}