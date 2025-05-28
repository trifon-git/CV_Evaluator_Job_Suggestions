import streamlit as st
import os
import traceback
from dotenv import load_dotenv
import PyPDF2
import docx
import markdown
import re
from datetime import datetime, timezone
import uuid
import csv
import pandas as pd
import plotly.express as px
import io
import tempfile
import time
from fpdf import FPDF # For PDF Generation

# Load environment variables
load_dotenv()

# --- Import your custom modules ---
try:
    from cv_match import find_similar_jobs, generate_embedding_for_skills, explain_job_match, get_embedding_model
    from extract_skills_from_cv_file import get_extracted_skills_from_file 
    from cover_letter_generator import CoverLetterGenerator
except ImportError as import_err:
    st.error(f"**Initialization Error:** Could not import custom modules: {import_err}")
    st.info("Ensure 'cv_match.py', 'extract_skills_from_cv_file.py', and 'cover_letter_generator.py' are in the app folder.")
    st.stop()
except Exception as general_import_err:
    st.error(f"**Initialization Error:** Unexpected error importing custom modules: {general_import_err}")
    traceback.print_exc()
    st.stop()


# Page config
st.set_page_config(
    page_title="CV Job Matcher | Denmark",
    page_icon="👔",
    layout="wide"
)

# Constants
MIN_CV_LENGTH_CHARS = 150
MAX_CV_LENGTH_CHARS = 20000
LOCAL_FEEDBACK_FILENAME = "feedback_log.csv" 
SIMILARITY_THRESHOLD = 40.0 
MAX_JOBS_TO_DISPLAY = 5
TOP_N_RESULTS_FROM_SEARCH = int(os.getenv('TOP_N_RESULTS', '20'))

# Initialize Cover Letter Generator
@st.cache_resource
def get_cover_letter_generator():
    try:
        return CoverLetterGenerator() 
    except Exception as e:
        st.error(f"Failed to initialize Cover Letter Generator: {e}. Check OPENAI_API key in .env.")
        return None

cover_letter_gen = get_cover_letter_generator()

# --- PDF Generation Function ---
class PDF(FPDF):
    def header(self):
        pass 
    def footer(self):
        pass 

def create_pdf_from_text(text_content):
    try:
        pdf = PDF()
        pdf.add_page()
        
        # Attempt to use a font that supports UTF-8.
        # If you have a .ttf file (e.g., DejaVuSans.ttf) in your app's directory:
        font_path = "DejaVuSans.ttf" # Example path, place the font file here or provide full path
        font_name = "DejaVu"
        try:
            if os.path.exists(font_path):
                 pdf.add_font(font_name, "", font_path, uni=True)
                 pdf.set_font(font_name, size=11)
                 print(f"Using font: {font_name} from {font_path}")
            else:
                raise RuntimeError(f"Font file {font_path} not found.")
        except RuntimeError as e_font:
            print(f"Warning: Custom font error ({e_font}). Falling back to core PDF fonts.")
            try:
                pdf.set_font("Arial", size=11)
                print("Using font: Arial")
            except RuntimeError:
                try:
                    pdf.set_font("Helvetica", size=11)
                    print("Using font: Helvetica")
                except RuntimeError:
                    pdf.set_font("Times", size=11) # Last resort
                    print("Using font: Times")
        
        # Use write() for better control over text and encoding with Unicode fonts
        # The h=5 is line height.
        # We need to handle the encoding of the text_content for write()
        # FPDF's write method expects text encoded in latin-1 by default for core fonts,
        # or correctly handled if a Unicode font (uni=True) is set.
        # Since we set uni=True for DejaVu, it should handle UTF-8 directly.
        # If falling back to core fonts, they have limited Unicode support.
        
        # Forcing UTF-8 and letting FPDF handle it with the selected font.
        # If text_content is already a str (Python 3 default), it's unicode.
        # No explicit encoding needed for pdf.write() if font is set up for unicode.
        
        pdf.write(5, text_content) # h=5 is line height.

        # Output to bytes
        pdf_output_bytes = pdf.output(dest='S')
        
        # FPDF2 `output(dest='S')` returns bytes.
        # If it were an older PyFPDF, it might return a string that needs encoding:
        # pdf_output_str = pdf.output(dest='S').encode('latin-1') # This was for older PyFPDF and core fonts
        
        if not pdf_output_bytes:
            st.error("PDF generation resulted in empty output. Check font compatibility and text content.")
            print("Error: PDF output bytes are empty.")
            return None
            
        return pdf_output_bytes

    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        print(f"Detailed PDF generation error: {traceback.format_exc()}")
        return None

# Function to read CV file in various formats (remains the same)
def read_cv_file(uploaded_file):
    if not uploaded_file:
        return None
    
    try:
        file_name = uploaded_file.name
        file_ext = os.path.splitext(file_name)[1].lower()
        cv_text = ""
        
        with st.status(f"Reading `{file_name}`...", expanded=False) as status:
            if file_ext == '.pdf':
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
                if not pdf_reader.pages:
                    status.update(label="Warning: PDF empty or unreadable.", state="warning")
                    return None
                for page_num in range(len(pdf_reader.pages)):
                    cv_text += (pdf_reader.pages[page_num].extract_text() or "") + "\n"
                if not cv_text.strip():
                    status.update(label="Warning: No text extracted from PDF.", state="warning")
            elif file_ext == '.docx':
                doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
                cv_text = "\n".join([p.text for p in doc.paragraphs if p.text])
                if not cv_text.strip():
                    status.update(label="Warning: No text found in DOCX.", state="warning")
            elif file_ext == '.md':
                md_bytes = uploaded_file.getvalue()
                try:
                    md_text = md_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    md_text = md_bytes.decode("latin-1", errors='ignore')
                html = markdown.markdown(md_text)
                cv_text = re.sub('<[^>]*>', ' ', html).strip()
                cv_text = re.sub(r'\s+', ' ', cv_text)
                if not cv_text.strip():
                    status.update(label="Warning: No text found in Markdown.", state="warning")
            elif file_ext == '.txt':
                txt_bytes = uploaded_file.getvalue()
                try:
                    cv_text = txt_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    cv_text = txt_bytes.decode("latin-1", errors='ignore')
                if not cv_text.strip():
                    status.update(label="Warning: Text file appears empty.", state="warning")
            else:
                st.error(f"Unsupported file type: `{file_ext}`.")
                status.update(label="Unsupported file type.", state="error")
                return None

            final_text = cv_text.strip()
            if final_text:
                status.update(label=f"Read `{file_name}`.", state="complete")
                return final_text
            else:
                if status.state != "warning" and status.state != "error":
                    status.update(label="Warning: No text content extracted from file.", state="warning")
                return None
    except Exception as e:
        st.error(f"Error reading file '{uploaded_file.name}'. Check format/corruption.")
        print(f"Error details reading file: {traceback.format_exc()}")
        return None

def initialize_local_feedback_csv():
    if not os.path.exists(LOCAL_FEEDBACK_FILENAME):
        try:
            with open(LOCAL_FEEDBACK_FILENAME, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "session_id", "cv_upload_time", "job_chroma_id", "rating", "cv_skills_count", "job_title_rated"])
            print(f"Created local feedback file: {LOCAL_FEEDBACK_FILENAME}")
        except Exception as e:
            st.error(f"Failed to create local feedback file: {e}")
            print(f"Error creating local feedback file: {e}")
            return False
    return True

def record_feedback_local(session_id, cv_upload_time, job_chroma_id, rating, cv_skills_count, job_title):
    if not initialize_local_feedback_csv():
        return False
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        new_row = [timestamp, session_id, cv_upload_time, job_chroma_id, rating, cv_skills_count, job_title]
        
        with open(LOCAL_FEEDBACK_FILENAME, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(new_row)
            
        print(f"Appended feedback locally to {LOCAL_FEEDBACK_FILENAME}")
        st.toast("Feedback saved!", icon="✅")
        return True
    except Exception as e:
        st.error(f"Error recording feedback locally: {e}", icon="❌")
        print(f"Error recording feedback: {traceback.format_exc()}")
        return False

def load_and_process_feedback():
    print("Loading and processing feedback from local CSV...")
    
    default_result = {
        "aggregates": {"per_job": {}, "total_up": 0, "total_down": 0},
        "dataframe": pd.DataFrame(columns=["timestamp", "session_id", "cv_upload_time", "job_chroma_id", "rating", "cv_skills_count", "job_title_rated"])
    }

    if not initialize_local_feedback_csv() or not os.path.isfile(LOCAL_FEEDBACK_FILENAME):
        print(f"Feedback file path invalid or missing: {LOCAL_FEEDBACK_FILENAME}")
        return default_result

    try:
        df = pd.read_csv(LOCAL_FEEDBACK_FILENAME, low_memory=False)
        per_job_feedback = {}
        total_up = 0
        total_down = 0

        if df.empty:
            print("Feedback CSV is empty")
            return default_result

        required_cols = ['rating', 'job_chroma_id', 'timestamp']
        if not all(col in df.columns for col in required_cols):
            print(f"Feedback CSV missing required columns")
            st.warning("Feedback file format incorrect. Stats might be incomplete.", icon="⚠️")
            valid_cols = [col for col in default_result["dataframe"].columns if col in df.columns]
            df_subset = df[valid_cols] if valid_cols else default_result["dataframe"]
            return {"aggregates": {"per_job": {}, "total_up": 0, "total_down": 0}, "dataframe": df_subset}

        total_counts = df['rating'].value_counts()
        total_up = int(total_counts.get('up', 0))
        total_down = int(total_counts.get('down', 0))

        try:
            job_counts = df.groupby('job_chroma_id')['rating'].value_counts().unstack(fill_value=0)
            if 'up' not in job_counts.columns: job_counts['up'] = 0
            if 'down' not in job_counts.columns: job_counts['down'] = 0
            per_job_feedback = job_counts.apply(lambda row: {"up": int(row['up']), "down": int(row['down'])}, axis=1).to_dict()
        except Exception as group_err:
            print(f"Error grouping feedback: {group_err}")
            st.warning("Could not calculate per-job feedback stats.", icon="⚠️")
            per_job_feedback = {}

        aggregates = {"per_job": per_job_feedback, "total_up": total_up, "total_down": total_down}

        df_processed = df.copy()
        df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], errors='coerce')
        df_processed.dropna(subset=['timestamp'], inplace=True)
        
        print(f"Loaded feedback: Up={total_up}, Down={total_down}, Rows={len(df_processed)}")
        return {"aggregates": aggregates, "dataframe": df_processed}

    except pd.errors.EmptyDataError:
        print(f"{LOCAL_FEEDBACK_FILENAME} is empty.")
        return default_result
    except Exception as e:
        print(f"Error processing feedback: {traceback.format_exc()}")
        st.error(f"Could not process feedback file. Stats may be unavailable.", icon="❌")
        return default_result

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'cv_upload_time' not in st.session_state:
    st.session_state.cv_upload_time = None
if 'feedback_given_jobs' not in st.session_state:
    st.session_state.feedback_given_jobs = {} 
if 'cv_skills' not in st.session_state:
    st.session_state.cv_skills = None
if 'generated_cover_letters' not in st.session_state:
    st.session_state.generated_cover_letters = {}


st.title("👨‍💼🇩🇰 CV Job Matcher")
st.markdown("Unlock your next career move in Denmark. Upload your CV and let our AI find jobs that **truly match your skills.**")
st.markdown("---")
st.markdown(f"""
**How It Works**
1.  You **upload your CV** (PDF, DOCX, TXT, MD).
2.  We **extract key skills** from your CV using AI.
3.  The system **compares your skill profile** to current, active job postings.
4.  You get the **top results**, ranked by match score (≥ {SIMILARITY_THRESHOLD:.0f}%).
5.  For each match, see **which of your skills contributed** most.
6.  Optionally, get an **AI-drafted cover letter** tailored to the job.
7.  Your feedback (👍/👎) is **saved locally** to help track system performance.

Ready to dive in?
""")
st.markdown("---")

if not all([os.getenv('EMBEDDING_API_URL'), os.getenv('CHROMA_HOST'), os.getenv('CHROMA_PORT'), os.getenv('CHROMA_COLLECTION')]):
    st.error("Missing critical backend settings in .env (EMBEDDING_API_URL, CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION). App cannot function.")
    st.stop()

if not os.getenv("OPENAI_API"):
    st.warning("OPENAI_API key not found in .env. Cover letter generation will be disabled.", icon="🔒")
    cover_letter_gen = None 

st.subheader("Upload Your CV 🚀")
uploaded_file = st.file_uploader("Choose CV file", type=['pdf', 'docx', 'txt', 'md'], 
                                label_visibility="collapsed", key="cv_uploader")

feedback_result = load_and_process_feedback()
feedback_aggregates = feedback_result["aggregates"]
feedback_df = feedback_result["dataframe"]

if uploaded_file is not None:
    file_process_placeholder = st.empty()
    file_process_placeholder.info(f"Processing `{uploaded_file.name}`...")
    
    st.session_state.cv_upload_time = datetime.now(timezone.utc).isoformat()
    st.session_state.feedback_given_jobs = {} 
    st.session_state.generated_cover_letters = {} 
    
    cv_text = read_cv_file(uploaded_file)
    
    if cv_text:
        text_length = len(cv_text)
        if not (MIN_CV_LENGTH_CHARS <= text_length <= MAX_CV_LENGTH_CHARS):
            file_process_placeholder.warning(f"CV Text Length: {text_length}. Recommended: {MIN_CV_LENGTH_CHARS}-{MAX_CV_LENGTH_CHARS} chars.", icon="📏")
        
        if text_length >= MIN_CV_LENGTH_CHARS:
            file_process_placeholder.success(f"CV Read Successfully ({text_length} chars). Extracting skills...")

            cv_skills = None
            with st.spinner("🤖 Extracting skills from your CV..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_cv_path = tmp_file.name
                    
                    cv_skills = get_extracted_skills_from_file(temp_cv_path)
                    os.unlink(temp_cv_path) 

                    if cv_skills:
                        st.session_state.cv_skills = cv_skills
                        file_process_placeholder.success(f"Extracted {len(cv_skills)} skills. Generating CV embedding...")
                        with st.expander("View Extracted CV Skills"):
                            st.write(cv_skills)
                    else:
                        file_process_placeholder.error("Could not extract skills from CV. Please check the CV content or try a different format.")
                        st.stop()
                except Exception as skill_ext_err:
                    file_process_placeholder.error(f"Error during CV skill extraction: {skill_ext_err}")
                    traceback.print_exc()
                    st.stop()
            
            cv_skill_embedding = None
            if cv_skills:
                with st.spinner("🧬 Generating embedding for your skills profile..."):
                    try:
                        cv_skill_embedding = generate_embedding_for_skills(cv_skills) 
                        if cv_skill_embedding is None:
                            file_process_placeholder.error("Could not generate embedding for CV skills.")
                            st.stop()
                        else:
                            file_process_placeholder.success("CV skills embedded. Searching for jobs...")
                    except Exception as embed_err:
                        file_process_placeholder.error(f"Error generating CV skill embedding: {embed_err}")
                        traceback.print_exc()
                        st.stop()
            
            display_matches = []
            all_matches_count = 0
            search_error = False

            if cv_skill_embedding is not None and cv_skills is not None:
                with st.spinner('🧠 Comparing CV to jobs... Please wait.'):
                    try:
                        all_job_matches, method_used = find_similar_jobs(
                            cv_skill_embedding=cv_skill_embedding, 
                            cv_skills=cv_skills, 
                            top_n=TOP_N_RESULTS_FROM_SEARCH, 
                            active_only=True
                        )

                        if all_job_matches is not None:
                            all_matches_count = len(all_job_matches)
                            filtered_matches = [j for j in all_job_matches if isinstance(j.get('score'), (int, float)) 
                                                and j.get('score', 0) >= SIMILARITY_THRESHOLD]
                            filtered_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
                            display_matches = filtered_matches[:MAX_JOBS_TO_DISPLAY]
                            print(f"Search method: {method_used}. Returned {all_matches_count} jobs. Filtered to {len(filtered_matches)}. Displaying {len(display_matches)}.")
                        else:
                            display_matches = []
                            all_matches_count = 0
                            st.warning(f"Search via {method_used} failed or returned no results.", icon="🤷")
                            if "Error" in method_used or "fail" in method_used.lower(): 
                                search_error = True
                    except Exception as find_err:
                        st.error(f"Error during job matching: {find_err}", icon="🔥")
                        traceback.print_exc()
                        display_matches = []
                        all_matches_count = 0
                        search_error = True
                file_process_placeholder.empty()

            tab_results, tab_feedback_analytics = st.tabs(["🎯 Matching Jobs", "📊 Overall Feedback"])

            with tab_results:
                if display_matches:
                    st.subheader(f"Top {len(display_matches)} Job Matches (Score ≥ {SIMILARITY_THRESHOLD:.0f}%)")
                    for i, job_match in enumerate(display_matches):
                        job_unique_id = job_match.get('chroma_id', f"job_fallback_{i}")
                        job_title = job_match.get('Title', 'N/A')
                        job_company = job_match.get('Company', 'N/A')
                        job_area = job_match.get('Area', 'N/A')
                        job_status = job_match.get('Status', 'unknown').capitalize()
                        job_url = job_match.get('url', '#')
                        job_score = job_match.get('score', 0.0)
                        contributing_skills = job_match.get('contributing_skills', [])
                        job_description_text = job_match.get('document_text', '') 

                        job_feedback_stats = feedback_aggregates["per_job"].get(job_unique_id, {"up": 0, "down": 0})

                        with st.container(border=True):
                            col_info, col_score_feedback = st.columns([3, 1.5])
                            with col_info:
                                st.markdown(f"**{i+1}. {job_title}**")
                                st.markdown(f"**Company:** {job_company} | **Location:** {job_area} | **Status:** `{job_status}`")
                                
                                with st.expander("See Matching Skills Analysis", expanded=False):
                                    if contributing_skills:
                                        st.markdown("Key CV skills contributing to this match:")
                                        for skill, skill_sim in contributing_skills:
                                            st.caption(f"- \"{skill}\" (Contribution Score: {skill_sim:.2f})")
                                    else:
                                        st.caption("Skill contribution analysis not available for this job.")
                                
                                action_cols = st.columns([1,1,2]) 
                                with action_cols[0]:
                                    if job_url and job_url != '#':
                                        st.link_button("Apply Now 🚀", url=job_url, type="primary", use_container_width=True)
                                    else:
                                        st.button("Apply Now", disabled=True, help="Application link not available", use_container_width=True)
                                with action_cols[1]:
                                    cl_button_key = f"cl_btn_{job_unique_id}"
                                    if cover_letter_gen and job_description_text:
                                        if st.button("Draft Cover Letter 📄", key=cl_button_key, use_container_width=True):
                                            with st.spinner("🖋️ Drafting cover letter... (this may take a moment)"):
                                                generated_letter = cover_letter_gen.generate_cover_letter(job_description_text, cv_text)
                                            if generated_letter and not generated_letter.lower().startswith("error:"):
                                                st.session_state.generated_cover_letters[job_unique_id] = generated_letter
                                            else:
                                                error_msg = generated_letter if generated_letter else "Error: Could not generate cover letter (empty response)."
                                                st.session_state.generated_cover_letters[job_unique_id] = error_msg
                                                st.error(error_msg) # Show error immediately if generation fails
                                    elif not job_description_text:
                                        st.button("Draft Cover Letter 📄", key=cl_button_key, disabled=True, help="Job description not available.", use_container_width=True)
                                    elif not cover_letter_gen:
                                        st.button("Draft Cover Letter 📄", key=cl_button_key, disabled=True, help="Cover letter generation disabled (OpenAI API key missing).", use_container_width=True)

                                if job_unique_id in st.session_state.generated_cover_letters:
                                    letter_content = st.session_state.generated_cover_letters[job_unique_id]
                                    if not letter_content.lower().startswith("error:"):
                                        st.markdown("**Generated Cover Letter Draft:**")
                                        st.text_area("", value=letter_content, height=400, # Increased height
                                                     key=f"cl_text_{job_unique_id}",
                                                     help="This is an AI-generated draft. Please review and edit carefully.")
                                        
                                        pdf_bytes = create_pdf_from_text(letter_content)
                                        if pdf_bytes:
                                            st.download_button(
                                                label="Download Cover Letter as PDF 💾",
                                                data=pdf_bytes,
                                                file_name=f"CoverLetter_{re.sub(r'[^a-zA-Z0-9_]', '', job_title)[:30]}.pdf", # Sanitize filename
                                                mime="application/pdf",
                                                key=f"dl_pdf_{job_unique_id}"
                                            )
                                        else:
                                            st.caption("Could not prepare PDF for download.")
                                    # Error message already shown when setting st.session_state if generation failed


                            with col_score_feedback:
                                st.metric("Match Score", f"{job_score:.1f}%", help="CV vs. Job similarity.")
                                st.write("Rate this match:")
                                
                                feedback_key_suffix = f"fb_{job_unique_id}_{st.session_state.cv_upload_time}"
                                current_rating = st.session_state.feedback_given_jobs.get(job_unique_id)

                                fb_buttons_cols = st.columns(2)
                                with fb_buttons_cols[0]:
                                    if st.button("👍", key=f"up_{feedback_key_suffix}", help="Good match", 
                                                 disabled=(current_rating is not None), use_container_width=True,
                                                 type="primary" if current_rating == "up" else "secondary"):
                                        if record_feedback_local(st.session_state.session_id, st.session_state.cv_upload_time, 
                                                                 job_unique_id, "up", len(st.session_state.cv_skills or []), job_title):
                                            st.session_state.feedback_given_jobs[job_unique_id] = "up"
                                            st.rerun() 
                                with fb_buttons_cols[1]:
                                    if st.button("👎", key=f"down_{feedback_key_suffix}", help="Not a good match", 
                                                 disabled=(current_rating is not None), use_container_width=True,
                                                 type="primary" if current_rating == "down" else "secondary"):
                                        if record_feedback_local(st.session_state.session_id, st.session_state.cv_upload_time, 
                                                                 job_unique_id, "down", len(st.session_state.cv_skills or []), job_title):
                                            st.session_state.feedback_given_jobs[job_unique_id] = "down"
                                            st.rerun() 
                                st.caption(f"Votes: 👍{job_feedback_stats.get('up', 0)} | 👎{job_feedback_stats.get('down', 0)}")
                                if current_rating:
                                    st.caption(f"✔️ You rated: {'👍' if current_rating == 'up' else '👎'}")
                
                elif not search_error and all_matches_count > 0 and not display_matches:
                    st.info(f"Found {all_matches_count} potential matches, but none scored {SIMILARITY_THRESHOLD:.0f}% or higher.", icon="📉")
                elif not search_error and all_matches_count == 0:
                    st.info("No relevant active jobs found matching your CV's skill profile.", icon="🤷")
                elif search_error:
                    st.error("The job search encountered an error. Please try again later or check logs.", icon="🔥")

            with tab_feedback_analytics: # This tab content remains the same
                st.subheader("Overall Match Performance & Feedback")
                st.markdown("This section provides insights into how well the job matcher is performing based on **all historical feedback** submitted by users like you (saved locally).")
                st.markdown("---")

                total_up = feedback_aggregates.get("total_up", 0)
                total_down = feedback_aggregates.get("total_down", 0)
                total_votes = total_up + total_down

                if total_votes > 0:
                    st.markdown("**Key Metrics**")
                    satisfaction_score = (total_up / total_votes) * 100 if total_votes > 0 else 0
                    col_t, col_s = st.columns(2)
                    with col_t: st.metric("Total Feedback Votes", total_votes)
                    with col_s: st.metric("Overall Satisfaction", f"{satisfaction_score:.1f}%")
                    st.markdown("---")

                    st.markdown("**Feedback Distribution**")
                    pie_data = pd.DataFrame({'Rating Type': ['Good Matches 👍', 'Bad Matches 👎'], 'Votes': [total_up, total_down]})
                    try:
                        fig_pie = px.pie(pie_data, values='Votes', names='Rating Type', color='Rating Type',
                                         color_discrete_map={'Good Matches 👍':'#2ECC71', 'Bad Matches 👎':'#E74C3C'})
                        fig_pie.update_layout(legend_title_text='Feedback Type', margin=dict(t=20,b=20,l=0,r=0))
                        fig_pie.update_traces(textposition='inside', textinfo='percent+value')
                        st.plotly_chart(fig_pie, use_container_width=True)
                    except Exception as plot_err: st.error(f"Failed to render pie chart: {plot_err}")
                    st.markdown("---")

                    st.markdown("**Feedback Trend Over Time (Daily)**")
                    if (feedback_df is not None and not feedback_df.empty and 
                        'timestamp' in feedback_df.columns and pd.api.types.is_datetime64_any_dtype(feedback_df['timestamp'])):
                        try:
                            daily_feedback = feedback_df.set_index('timestamp').resample('D')['rating'].value_counts().unstack(fill_value=0)
                            if 'up' not in daily_feedback.columns: daily_feedback['up'] = 0
                            if 'down' not in daily_feedback.columns: daily_feedback['down'] = 0
                            daily_feedback = daily_feedback.rename(columns={'up': 'Good 👍', 'down': 'Bad 👎'})
                            if not daily_feedback.empty and (daily_feedback['Good 👍'].sum() > 0 or daily_feedback['Bad 👎'].sum() > 0):
                                fig_time = px.line(daily_feedback, y=['Good 👍', 'Bad 👎'], markers=True,
                                                   labels={"timestamp": "Date", "value": "Number of Ratings", "variable": "Rating Type"},
                                                   color_discrete_map={'Good 👍': '#2ECC71', 'Bad 👎': '#E74C3C'})
                                fig_time.update_layout(hovermode="x unified", legend_title_text='Rating Type', yaxis_title="Number of Ratings")
                                st.plotly_chart(fig_time, use_container_width=True)
                            else: st.info("Not enough feedback data points to display a trend.", icon="📈")
                        except Exception as time_plot_err: 
                            st.error(f"Failed to render trend chart: {time_plot_err}")
                            print(f"Error rendering time series: {traceback.format_exc()}")
                    else: st.info("Not enough valid timestamp data for trend analysis.", icon="⏳")
                else:
                    st.info("📊 No feedback votes recorded yet. Rate matches to build these statistics!", icon="✏️")
    else:
        if uploaded_file: 
            file_process_placeholder.error("Could not read CV content. Please try a different file or format.")

st.divider()
st.caption("CV Job Matcher | MLOps Project Demo")
