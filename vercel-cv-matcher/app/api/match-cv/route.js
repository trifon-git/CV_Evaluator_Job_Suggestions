import { NextResponse } from 'next/server';
// Change this import path to use a relative path instead of the alias
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
    const cvText = buffer.toString('utf-8');
    
    // Process the CV using our matching algorithm
    const results = await processCV(cvText);
    
    return NextResponse.json({ results });
  } catch (error) {
    console.error('Error processing CV:', error);
    return NextResponse.json(
      { error: 'Failed to process CV' },
      { status: 500 }
    );
  }
}