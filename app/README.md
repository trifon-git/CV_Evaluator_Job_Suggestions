# 🇩🇰 CV Job Matcher - Denmark

An AI app to match your CV with relevant job postings in Denmark.

**[▶️ Try it live on Hugging Face Spaces](https://huggingface.co/spaces/Lpiziks2/CV_Evaluator_Job_Suggestions)**

## How It Works

The system uses advanced natural language processing to match CVs with job listings:

1. **Skill Extraction**: Extracts skills from your CV using AI
2. **Multi-Dimensional Matching**: 
   - Creates embeddings for individual skills and the complete CV
   - Matches against job postings using vector similarity
   - Performs direct skill-to-skill matching
   - Uses bidirectional alignment to understand skill importance

3. **Smart Filtering**: Filter jobs by location, category, and required languages

4. **Cover Letter Generation**: Creates customized cover letters for your target jobs

## Technology Stack

- **Streamlit**: Interactive web interface
- **Vector Embeddings**: Semantic understanding using sentence transformers
- **ChromaDB**: Vector database for efficient semantic search
- **LLM Processing**: Skill extraction and cover letter generation
