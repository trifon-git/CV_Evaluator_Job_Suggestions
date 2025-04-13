import { NextResponse } from 'next/server';
import { processCV } from '../../../lib/cv-processor';

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
    let cvText;
    
    try {
      cvText = buffer.toString('utf-8');
      console.log(`CV text loaded (${cvText.length} characters)`);
    } catch (error) {
      console.error('Error reading file content:', error);
      return NextResponse.json(
        { error: 'Failed to read CV file content' },
        { status: 500 }
      );
    }
    
    // Process the CV using our matching algorithm
    try {
      const results = await processCV(cvText);
      return NextResponse.json({ results });
    } catch (error) {
      console.error('Error in CV processing:', error);
      return NextResponse.json(
        { error: `Failed to process CV: ${error.message}` },
        { status: 500 }
      );
    }
  } catch (error) {
    console.error('Error processing CV request:', error);
    return NextResponse.json(
      { error: 'Failed to process CV request' },
      { status: 500 }
    );
  }
}