# 🇩🇰 CV Job Matcher - Denmark

An AI-powered application that matches your CV with relevant job postings in Denmark by analyzing skills, experience, and job requirements.

**[▶️ Try it live on Hugging Face Spaces](https://huggingface.co/spaces/Krepselis/CV_Evalutaor_Job_Suggestions)**

## 📋 Project Overview

CV Job Matcher uses advanced natural language processing and AI to help job seekers in Denmark find positions that match their skills and experience. The application analyzes your CV, extracts relevant skills, and compares them against current job listings to identify the best matches.

## 🚀 How It Works

The system leverages advanced natural language processing to match CVs with job listings:

1. **Skill Extraction**: 
   - Automatically extracts skills, experience, and qualifications from your CV using AI
   - Identifies both explicit and implicit skills from your work history

2. **Multi-Dimensional Matching**: 
   - Creates embeddings for individual skills and the complete CV
   - Matches against job postings using vector similarity
   - Performs direct skill-to-skill matching
   - Uses bidirectional alignment to understand skill importance and relevance

3. **Smart Filtering**: 
   - Filter jobs by location within Denmark
   - Filter by job category or industry
   - Filter by required languages (Danish, English, etc.)
   - Adjust matching preferences to prioritize different aspects

4. **Cover Letter Generation**: 
   - Creates customized cover letters for your target jobs
   - Highlights relevant skills and experience
   - Tailors content to specific job requirements

