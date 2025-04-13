import streamlit as st
import os
import traceback 
from dotenv import load_dotenv

# Import file reading libraries
import PyPDF2
import docx
import markdown
import re 

try:
    from cv_match import find_similar_jobs
except ImportError:
    # This is a critical error, stop the app
    st.error(
        "🚨 **Initialization Error:** Could not import `find_similar_jobs` from `cv_match.py`. "
        "Please ensure the file exists in the `src` directory and has no syntax errors."
    )
    st.stop()

load_dotenv()

def read_cv_file(uploaded_file):
    """Reads text content from uploaded file (PDF, DOCX, TXT, MD)."""
    try:
        file_name = uploaded_file.name
        file_ext = os.path.splitext(file_name)[1].lower()
        cv_text = ""

        # Use a status placeholder while reading
        read_status = st.empty()
        read_status.info(f"⏳ Reading file: {file_name}...")

        if file_ext == '.pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            if not pdf_reader.pages:
                 st.warning("⚠️ The PDF seems to be empty or corrupted.")
                 read_status.empty()
                 return None
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        cv_text += page_text + "\n"
                except Exception as page_err:
                    st.warning(f"⚠️ Could not extract text from PDF page {page_num + 1}: {page_err}")
            if not cv_text:
                 st.warning("⚠️ No text could be extracted from the PDF. It might be image-based or protected.")

        elif file_ext == '.docx':
            doc = docx.Document(uploaded_file)
            cv_text = "\n".join([para.text for para in doc.paragraphs if para.text])
            if not cv_text:
                 st.warning("⚠️ No text could be extracted from the DOCX file.")

        elif file_ext == '.md':
            md_bytes = uploaded_file.read()
            try:
                md_text = md_bytes.decode("utf-8")
            except UnicodeDecodeError:
                st.warning("⚠️ Could not decode markdown as UTF-8, trying latin-1.")
                md_text = md_bytes.decode("latin-1")
            html_text = markdown.markdown(md_text)
            cv_text = re.sub('<[^<]+?>', ' ', html_text).strip()
            if not cv_text:
                 st.warning("⚠️ No text could be extracted after processing the Markdown file.")

        elif file_ext == '.txt':
            txt_bytes = uploaded_file.read()
            try:
                cv_text = txt_bytes.decode("utf-8")
            except UnicodeDecodeError:
                st.warning("⚠️ Could not decode text file as UTF-8, trying latin-1.")
                cv_text = txt_bytes.decode("latin-1")
            if not cv_text:
                st.warning("⚠️ The text file appears to be empty.")

        else:
            st.error(f"Unsupported file type: {file_ext}")
            read_status.empty()
            return None

        # Clear the reading status message
        read_status.empty()

        final_text = cv_text.strip()
        if not final_text:
             st.warning(f"Extracted text from {file_name} appears to be empty after processing.")
             return None # Return None if no usable text was found

        st.success(f"✅ Successfully read content from `{file_name}`.")
        return final_text

    except Exception as e:
        st.error(f"An error occurred while reading the file '{uploaded_file.name}':")
        st.exception(e)
        if 'read_status' in locals(): read_status.empty() # Clear status on error
        return None

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="CV Job Matcher | Denmark",
    page_icon="👔",
    layout="wide"
)

# --- App Header ---
st.title("🇩🇰 CV Job Matcher: Denmark")
st.markdown(
    "Upload your CV to discover relevant Danish job postings from our database. "
    "We analyze your CV's content and compare it against active job descriptions."
)

# --- How it Works Section ---
with st.expander("How it Works & Understanding the Score"):
    st.markdown(
        """
        1.  **Upload:** You provide your CV (PDF, DOCX, TXT, or MD).
        2.  **Extract:** We extract the text content from your CV.
        3.  **Embed:** The text is converted into a numerical representation (embedding) that captures its semantic meaning using an AI model.
        4.  **Compare:** This CV embedding is compared against the embeddings of active job descriptions stored in our database (ChromaDB).
        5.  **Rank:** Jobs are ranked based on the similarity between their description's embedding and your CV's embedding.

        **Match Score:**
        *   The score indicates the **semantic similarity** between your CV and the job description, based on the vector comparison. A higher score means the topics and concepts discussed are closer.
        *   It's calculated using vector distance (lower distance = higher similarity). We convert this distance into a percentage for easier interpretation.
        *   **Important:** This score reflects *content similarity*, not necessarily a perfect qualification match. It's a starting point to help you identify potentially relevant roles. Always review the full job description carefully.
        """
    )

st.divider()

# --- Configuration Check ---
# (Keep the configuration check section as before)
embedding_api_url = os.getenv('EMBEDDING_API_URL')
chroma_host = os.getenv('CHROMA_HOST')
config_ok = True
if not embedding_api_url:
    st.error("🚨 **Configuration Error:** `EMBEDDING_API_URL` is not set. Cannot generate CV embeddings.")
    config_ok = False
if not chroma_host:
    st.warning("⚠️ **Configuration Note:** `CHROMA_HOST` is not set. Assuming ChromaDB is accessible.")
if not config_ok:
    st.stop()

# --- File Upload Section ---
st.subheader("1. Upload Your CV")
uploaded_file = st.file_uploader(
    "Choose your CV file (PDF, DOCX, TXT, MD)", # Label made more concise
    type=['pdf', 'docx', 'txt', 'md'],
    help="Upload your CV. Max file size 200MB.",
    label_visibility="collapsed", # Hide the label above as we use st.subheader
    key="cv_uploader"
)

# --- Main Processing Logic ---
if uploaded_file is not None:

    # No more column indentation here
    st.info(f"📄 Processing file: `{uploaded_file.name}` ({uploaded_file.size / 1024:.1f} KB)")

    cv_text = read_cv_file(uploaded_file)

    if cv_text:
        st.subheader("2. Finding Matches...")
        with st.spinner('🧠 Analyzing CV content & searching database... Please wait...'):
            try:
                matches, method_used = find_similar_jobs(cv_text, active_only=True)

                # --- Display Results ---
                st.subheader("3. Matching Jobs Found")

                if matches is not None and len(matches) > 0:
                    st.success(f"✨ Found {len(matches)} potential job matches!")

                    for i, job in enumerate(matches):
                        with st.container():
                            st.divider()
                            job_title = job.get('Title', 'N/A')
                            job_url = job.get('url', '#')
                            job_score = job.get('score', 0)
                            job_company = job.get('Company', 'N/A')
                            job_area = job.get('Area', 'N/A')
                            job_status = job.get('Status', 'N/A')
                            job_content = job.get('content', '')

                            st.markdown(f"#### {i+1}. [{job_title}]({job_url})")

                            detail_col1, detail_col2, detail_col3 = st.columns([1.5, 3, 1.5])
                            with detail_col1:
                                st.metric("Match Score", f"{job_score:.1f}%")
                            with detail_col2:
                                st.markdown(f"🏢 **Company:** {job_company}")
                                st.markdown(f"📍 **Area:** {job_area}")
                            with detail_col3:
                                 st.markdown(f"📊 **Status:** `{job_status.capitalize()}`")

                            if job_content:
                                with st.expander("Show Job Content Snippet"):
                                    st.text(job_content[:500] + ("..." if len(job_content) > 500 else ""))

                elif matches is not None and len(matches) == 0:
                    st.info("ℹ️ No matching active jobs were found based on your CV's content. You might refine your CV or check back later as new jobs are added.")
                elif method_used and method_used.startswith("Error"):
                    st.error(f"⚠️ **Search Failed:** {method_used}")
                    st.error("Could not retrieve job matches.")
                else:
                    st.error("❓ An unexpected issue occurred during the job search.")

            except Exception as find_err:
                 st.error("❌ **Critical Error:** An error occurred during the job matching process.")
                 st.exception(find_err)
    else:
        st.error("⚠️ Could not extract readable text from the uploaded file. Please check the file and try again.")

# --- Footer ---
st.divider()
st.caption("CV Job Matcher | Powered by Streamlit & AI")

