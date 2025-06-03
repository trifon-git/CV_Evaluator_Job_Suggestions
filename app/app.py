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
from fpdf import FPDF
import json

# Load environment variables
load_dotenv()

# --- Import your custom modules ---
try:
    from cv_match import find_similar_jobs, generate_embedding_for_skills
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
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants and Normalization Data ---
MIN_CV_LENGTH_CHARS = 150
MAX_CV_LENGTH_CHARS = 20000
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_FEEDBACK_FILENAME = os.path.join(APP_DIR, "feedback_log.csv")

# --- AGGRESSIVE DEBUGGING (Remove or comment out after fixing) ---
print("--- STARTUP DEBUG ---")
print(f"app.py __file__ (relative to execution): {__file__}")
print(f"APP_DIR resolved to: {APP_DIR}")
print(f"LOCAL_FEEDBACK_FILENAME resolved to: {LOCAL_FEEDBACK_FILENAME}")
print(f"Does LOCAL_FEEDBACK_FILENAME exist? {os.path.exists(LOCAL_FEEDBACK_FILENAME)}")
print(f"Is LOCAL_FEEDBACK_FILENAME a file? {os.path.isfile(LOCAL_FEEDBACK_FILENAME)}")
if os.path.exists(LOCAL_FEEDBACK_FILENAME):
    try:
        with open(LOCAL_FEEDBACK_FILENAME, 'r', encoding='utf-8') as f_debug:
            print(f"First 100 chars of feedback file: {f_debug.read(100)}")
    except Exception as e_debug_read:
        print(f"Error trying to read first few chars: {e_debug_read}")
else:
    print(f"NOTE: {LOCAL_FEEDBACK_FILENAME} does not exist. Will be created by initialize_local_feedback_csv.")
print("--- END STARTUP DEBUG ---")
# --- END AGGRESSIVE DEBUGGING ---

SIMILARITY_THRESHOLD = 10
MAX_JOBS_TO_DISPLAY_PER_PAGE = 5
TOP_N_RESULTS_FROM_SEARCH = int(os.getenv('TOP_N_RESULTS_FOR_APP_QUERY', '100000'))
CANONICAL_LANGUAGES_FOR_FILTER = ["English", "Danish", "German", "Spanish", "French", "Norwegian", "Swedish"]

# --- Helper Functions ---
def get_job_languages_from_metadata(metadata_dict):
    extracted_languages_set = set()
    
    # Method 1: Parse JSON language requirements
    lang_req_json_str = metadata_dict.get("Language_Requirements_Json_Str")
    if lang_req_json_str:
        try:
            lang_data = json.loads(lang_req_json_str)
            if isinstance(lang_data, list):
                for lang_entry in lang_data:
                    # Handle both formats:
                    # 1. String format: ["Danish", "English"]
                    if isinstance(lang_entry, str) and lang_entry.strip():
                        lang_name = lang_entry.strip()
                        for canonical in CANONICAL_LANGUAGES_FOR_FILTER:
                            if canonical.lower() == lang_name.lower():
                                extracted_languages_set.add(canonical)
                                break
                    # 2. Object format: [{"language": "Danish", "proficiency": "Fluent"}]
                    elif isinstance(lang_entry, dict):
                        lang_name = lang_entry.get("language")
                        if isinstance(lang_name, str) and lang_name.strip():
                            for canonical in CANONICAL_LANGUAGES_FOR_FILTER:
                                if canonical.lower() == lang_name.strip().lower():
                                    extracted_languages_set.add(canonical)
                                    break
        except Exception as e:
            print(f"Error parsing Language_Requirements_Json_Str: {e}")
            pass
    
    # Method 2: Check for proficiency fields (no change needed)
    for key, value in metadata_dict.items():
        if key.startswith("lang_") and key.endswith("_proficiency"):
            lang_name_from_key = key.replace("lang_", "").replace("_proficiency", "")
            for canonical in CANONICAL_LANGUAGES_FOR_FILTER:
                if canonical.lower() == lang_name_from_key.lower():
                    extracted_languages_set.add(canonical)
                    break
    
    # Method 3: Use detected ad language (no change needed)
    detected_ad_lang_raw = metadata_dict.get("Detected_Ad_Language")
    if isinstance(detected_ad_lang_raw, str) and detected_ad_lang_raw.strip() and detected_ad_lang_raw != "Unknown":
        for canonical in CANONICAL_LANGUAGES_FOR_FILTER:
            if canonical.lower() == detected_ad_lang_raw.strip().lower():
                extracted_languages_set.add(canonical)
                break
    
    # Method 4: Check for raw language_requirements array directly in metadata
    raw_lang_reqs = metadata_dict.get("language_requirements")
    if isinstance(raw_lang_reqs, list):
        for lang_item in raw_lang_reqs:
            if isinstance(lang_item, str) and lang_item.strip():
                for canonical in CANONICAL_LANGUAGES_FOR_FILTER:
                    if canonical.lower() == lang_item.strip().lower():
                        extracted_languages_set.add(canonical)
                        break
    
    return sorted(list(extracted_languages_set))

@st.cache_resource
def get_cover_letter_generator():
    try: return CoverLetterGenerator()
    except Exception as e: st.error(f"Cover Letter Generator Error: {e}. Check OPENAI_API_KEY."); return None
cover_letter_gen = get_cover_letter_generator()

class PDF(FPDF):
    def header(self): pass
    def footer(self): pass

def create_pdf_from_text(text_content):
    try:
        pdf = PDF(); pdf.add_page()
        font_path = os.path.join(APP_DIR, "DejaVuSans.ttf")
        font_name = "DejaVu"
        try:
            if os.path.exists(font_path): pdf.add_font(font_name, "", font_path, uni=True)
            else: raise RuntimeError(f"Font file {font_path} not found. Ensure it's in: {APP_DIR}")
            pdf.set_font(font_name, size=11)
        except RuntimeError:
            print(f"Warning: Custom font {font_name} at {font_path} not found. Falling back to Arial.")
            try: pdf.set_font("Arial", size=11)
            except RuntimeError: pdf.set_font("Helvetica", size=11)
        pdf.write(5, text_content); pdf_output_bytes = pdf.output(dest='S')
        if not pdf_output_bytes: st.error("PDF generation resulted in empty output."); return None
        return pdf_output_bytes
    except Exception as e: st.error(f"Error generating PDF: {e}"); print(f"PDF error: {traceback.format_exc()}"); return None

def read_cv_file(uploaded_file):
    if not uploaded_file: return None
    try:
        file_name=uploaded_file.name; file_ext=os.path.splitext(file_name)[1].lower(); cv_text=""
        with st.spinner(f"Reading `{file_name}`..."):
            content_bytes = uploaded_file.getvalue()
            if file_ext=='.pdf': r=PyPDF2.PdfReader(io.BytesIO(content_bytes)); cv_text="".join([(p.extract_text() or "") for p in r.pages])
            elif file_ext=='.docx': d=docx.Document(io.BytesIO(content_bytes)); cv_text="\n".join([p.text for p in d.paragraphs])
            elif file_ext=='.md': h=markdown.markdown(content_bytes.decode("utf-8",errors="ignore")); cv_text=re.sub('<[^>]*>',' ',h).strip()
            elif file_ext=='.txt': cv_text=content_bytes.decode("utf-8",errors="ignore")
            else: st.error(f"Unsupported: `{file_ext}`."); return None
        final_text=cv_text.strip()
        if not final_text: st.warning("No text extracted."); return None
        return final_text
    except Exception as e: st.error(f"Error reading '{uploaded_file.name}'."); print(f"Read error: {traceback.format_exc()}"); return None

# --- ADJUSTED initialize_local_feedback_csv ---
def initialize_local_feedback_csv():
    if not os.path.exists(LOCAL_FEEDBACK_FILENAME):
        try:
            # Header matches your CURRENT actual CSV structure
            header = ["timestamp", "session_id", "cv_upload_time", "job_chroma_id", 
                      "predicted_score", "rank_displayed", "rating"] 
            with open(LOCAL_FEEDBACK_FILENAME, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(header)
            print(f"INFO: Initialized {LOCAL_FEEDBACK_FILENAME} with header reflecting actual CSV: {header}")
        except Exception as e:
            st.error(f"Feedback file initialization error: {e}")
            return False
    return True

# --- ADJUSTED record_feedback_local ---
def record_feedback_local(sid, cv_ts, jid, rt, predicted_score_val, rank_displayed_val): 
    if not initialize_local_feedback_csv(): return False
    try:
        feedback_data = [
            datetime.now(timezone.utc).isoformat(), # timestamp
            sid,                                     # session_id
            cv_ts,                                   # cv_upload_time
            jid,                                     # job_chroma_id
            predicted_score_val if predicted_score_val is not None else "", # predicted_score
            rank_displayed_val if rank_displayed_val is not None else "",   # rank_displayed
            rt                                       # rating ('up' or 'down')
        ]
        with open(LOCAL_FEEDBACK_FILENAME, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(feedback_data)
        st.toast("Feedback saved!", icon="👍"); return True
    except Exception as e: st.error(f"Feedback record error: {e}"); return False

# --- ADJUSTED load_and_process_feedback ---
def load_and_process_feedback():
    print(f"DEBUG: load_and_process_feedback called. Trying to load: {LOCAL_FEEDBACK_FILENAME}")
    # Internal names can be whatever you prefer, but map from actual CSV
    default_columns = ["ts", "sid", "cv_ts", "jid", "pred_score", "rank_disp", "rt"] 
    def_res = {"aggregates":{"per_job":{},"total_up":0,"total_down":0},"dataframe":pd.DataFrame(columns=default_columns)}

    if not os.path.isfile(LOCAL_FEEDBACK_FILENAME):
        print(f"DEBUG: load_and_process_feedback - {LOCAL_FEEDBACK_FILENAME} is NOT a file or does not exist. Returning default.")
        return def_res

    print(f"DEBUG: load_and_process_feedback - {LOCAL_FEEDBACK_FILENAME} IS a file. Attempting to read with pandas.")
    try:
        df = pd.read_csv(LOCAL_FEEDBACK_FILENAME, low_memory=False, encoding='utf-8')
        
        print(f"DEBUG: load_and_process_feedback - Pandas read_csv successful.")
        print(f"DEBUG: DataFrame shape: {df.shape}")
        print(f"DEBUG: DataFrame columns FROM CSV: {df.columns.tolist()}")

        if df.empty and df.shape[0] == 0:
            print(f"DEBUG: load_and_process_feedback - DataFrame is empty after read (no data rows, possibly just header). Returning default.")
            return def_res

        column_rename_map = {
            "timestamp": "ts",
            "session_id": "sid",
            "cv_upload_time": "cv_ts",
            "job_chroma_id": "jid",
            "predicted_score": "pred_score", # Maps from actual CSV column
            "rank_displayed": "rank_disp", # Maps from actual CSV column
            "rating": "rt"
        }

        actual_csv_columns = df.columns.tolist()
        missing_from_csv = [csv_col for csv_col in column_rename_map.keys() if csv_col not in actual_csv_columns]
        if missing_from_csv:
            print(f"WARNING: The CSV file '{LOCAL_FEEDBACK_FILENAME}' is missing columns needed for renaming: {missing_from_csv}")
            print(f"         Actual columns found in CSV: {actual_csv_columns}")
            # For these specific missing columns, it's okay to proceed if others are fine
            # st.warning(f"Feedback CSV is missing columns: {', '.join(missing_from_csv)}. Analytics may be incomplete.")

        df = df.rename(columns=column_rename_map)
        print(f"DEBUG: DataFrame columns AFTER RENAME: {df.columns.tolist()}")

        internal_required_cols_for_aggregation = ['rt', 'jid', 'ts']
        missing_internal_cols = [col for col in internal_required_cols_for_aggregation if col not in df.columns]
        if missing_internal_cols:
            print(f"ERROR: After renaming, internal required columns for aggregation are missing: {missing_internal_cols}")
            st.warning(f"Feedback data processing error: Critical columns missing ({', '.join(missing_internal_cols)}). Analytics incomplete.")
            return def_res

        tu = int(df['rt'].value_counts().get('up', 0))
        td = int(df['rt'].value_counts().get('down', 0))
        
        pj = {}
        try:
            if 'jid' in df.columns and 'rt' in df.columns:
                jc = df.groupby('jid')['rt'].value_counts().unstack(fill_value=0)
                if 'up' not in jc.columns: jc['up'] = 0
                if 'down' not in jc.columns: jc['down'] = 0
                pj = jc.apply(lambda r: {"up": int(r['up']), "down": int(r['down'])}, axis=1).to_dict()
            else: print("DEBUG: 'jid' or 'rt' missing after rename, skipping per-job aggregation.")
        except Exception as e_pj: print(f"DEBUG: Error during per-job aggregation: {e_pj}")

        aggs = {"per_job": pj, "total_up": tu, "total_down": td}
        
        if 'ts' in df.columns:
            df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
            df.dropna(subset=['ts'], inplace=True)
        else: print("DEBUG: 'ts' column (timestamp) missing after rename, skipping timestamp processing.")

        print(f"DEBUG: load_and_process_feedback - Aggregates: {aggs}")
        return {"aggregates": aggs, "dataframe": df}

    except FileNotFoundError:
        print(f"DEBUG: load_and_process_feedback - Explicit FileNotFoundError for {LOCAL_FEEDBACK_FILENAME}. Returning default.")
        return def_res
    except pd.errors.EmptyDataError:
        print(f"DEBUG: load_and_process_feedback - Pandas EmptyDataError for {LOCAL_FEEDBACK_FILENAME} (file is empty or has only headers). Returning default.")
        return def_res
    except Exception as e:
        print(f"DEBUG: load_and_process_feedback - Other exception during pandas read or processing for {LOCAL_FEEDBACK_FILENAME}: {e}")
        traceback.print_exc()
        st.warning(f"Could not process feedback file: {e}")
        return def_res
# --- END ADJUSTED load_and_process_feedback ---

# --- Streamlit App State Initialization ---
if 'session_id' not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
if 'cv_upload_time' not in st.session_state: st.session_state.cv_upload_time = None
if 'feedback_given_jobs' not in st.session_state: st.session_state.feedback_given_jobs = {}
if 'cv_skills' not in st.session_state: st.session_state.cv_skills = None
if 'generated_cover_letters' not in st.session_state: st.session_state.generated_cover_letters = {}
if 'all_job_matches_cache' not in st.session_state: st.session_state.all_job_matches_cache = None
if 'cv_text_cache' not in st.session_state: st.session_state.cv_text_cache = None

# --- App Header & Intro ---
st.title("👨‍💼🇩🇰 CV Job Matcher")
st.subheader("Unlock Your Next Career Move in Denmark!")
st.markdown("Upload your CV, and our AI will find jobs that truly match your skills, help you understand why, and even draft a cover letter.")
st.markdown("---")

# --- Prerequisite Checks ---
if not all([os.getenv('EMBEDDING_API_URL'), os.getenv('CHROMA_HOST'), os.getenv('CHROMA_PORT'), os.getenv('CHROMA_COLLECTION')]):
    st.error("Backend services are not fully configured. Please check `.env` settings. The app cannot function without them.")
    st.stop()
if not os.getenv("OPENAI_API_KEY"):
    st.warning("OpenAI API key not found. Cover letter generation will be disabled.", icon="🔒")
    cover_letter_gen = None

# --- Main App Layout ---
with st.sidebar:
    st.header("🚀 Get Started")
    uploaded_file = st.file_uploader("1. Upload Your CV", type=['pdf', 'docx', 'txt', 'md'],
                                    key="cv_uploader_key",
                                    help="Supports PDF, DOCX, Markdown, and TXT files.",
                                    on_change=lambda: st.session_state.update(all_job_matches_cache=None, generated_cover_letters={}, cv_text_cache=None, cv_skills=None))
    st.markdown("---")
    st.header("🔍 Filter Job Matches")
    unique_locations_options = []
    unique_categories_options = []
    if st.session_state.all_job_matches_cache:
        # Fix: Use lowercase field names that match cv_match.py output
        unique_locations_options = sorted(list(set(job.get('area') for job in st.session_state.all_job_matches_cache if job.get('area'))))
        raw_categories_from_cache = [job.get('category') for job in st.session_state.all_job_matches_cache]
        unique_categories_options = sorted(list(set(cat for cat in raw_categories_from_cache if isinstance(cat, str) and cat.strip())))
        
        if not unique_categories_options and any(st.session_state.all_job_matches_cache):
            sample_has_category_key = False
            for job_sample in st.session_state.all_job_matches_cache[:min(3, len(st.session_state.all_job_matches_cache))]:
                if 'category' in job_sample: sample_has_category_key = True; break  # Fix: lowercase 'category'
            if not sample_has_category_key: st.sidebar.caption("⚠️ 'category' field missing from data.")
            else: st.sidebar.caption("⚠️ No filterable categories in data.")
    selected_locations = st.multiselect("Job Locations (Area)", options=unique_locations_options, placeholder="Any Location" if unique_locations_options else "Upload CV to see locations")
    selected_categories = st.multiselect("Job Categories", options=unique_categories_options, placeholder="Any Category" if unique_categories_options else "Upload CV to see categories")
    selected_languages = st.multiselect("Required Languages", options=CANONICAL_LANGUAGES_FOR_FILTER, placeholder="Any Language")
    st.markdown("---")
    st.info("Tip: Upload your CV first, then apply filters to the results.")
    st.caption(f"Matching up to {TOP_N_RESULTS_FROM_SEARCH} jobs, displaying top {MAX_JOBS_TO_DISPLAY_PER_PAGE} after filters.")

placeholder_processing_status = st.empty()
feedback_result = load_and_process_feedback()
feedback_aggregates = feedback_result["aggregates"]
feedback_df = feedback_result["dataframe"]

if uploaded_file is not None:
    if st.session_state.all_job_matches_cache is None:
        with placeholder_processing_status.container():
            with st.spinner(f"Analyzing `{uploaded_file.name}`... This might take a few moments."):
                st.session_state.cv_upload_time = datetime.now(timezone.utc).isoformat()
                st.session_state.feedback_given_jobs = {}
                st.session_state.generated_cover_letters = {}
                cv_text = read_cv_file(uploaded_file)
                st.session_state.cv_text_cache = cv_text
                if cv_text:
                    with st.spinner("🤖 Extracting skills from your CV..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue()); temp_cv_path = tmp_file.name
                        cv_skills_list = get_extracted_skills_from_file(temp_cv_path)
                        if os.path.exists(temp_cv_path): os.unlink(temp_cv_path)
                    if not cv_skills_list: placeholder_processing_status.error("Could not extract skills."); st.stop()
                    st.session_state.cv_skills = cv_skills_list
                    if len(cv_skills_list) < 50:
                         with st.expander("💡 View Extracted CV Skills", expanded=False): st.caption(f"{len(cv_skills_list)} skills: {', '.join(cv_skills_list)}")
                    with st.spinner("🧬 Generating CV skills embedding..."):
                        cv_skill_embedding_vec = generate_embedding_for_skills(cv_skills_list)
                    if cv_skill_embedding_vec is None: placeholder_processing_status.error("Could not generate CV embedding."); st.stop()
                    with st.spinner('🧠 Searching for matching jobs...'):
                        all_matches_db = find_similar_jobs(cv_skills=cv_skills_list, cv_embedding=cv_skill_embedding_vec, top_n=TOP_N_RESULTS_FROM_SEARCH, filter_active_only=True)
                        st.session_state.all_job_matches_cache = all_matches_db if all_matches_db is not None else []
                else: placeholder_processing_status.error("Could not read CV content."); st.session_state.all_job_matches_cache = []
            if st.session_state.all_job_matches_cache is not None:
                placeholder_processing_status.success(f"CV processing complete! Found {len(st.session_state.all_job_matches_cache)} potential matches. Filters updated.")
                time.sleep(1); st.rerun()
            else: placeholder_processing_status.warning("CV processing finished, but no initial matches found or an error occurred.")
            time.sleep(2); placeholder_processing_status.empty()

if st.session_state.all_job_matches_cache is not None:
    current_matches_to_filter = list(st.session_state.all_job_matches_cache)
    # Fix: Use lowercase field names
    if selected_locations: 
        current_matches_to_filter = [job for job in current_matches_to_filter if job.get('area') in selected_locations]
    if selected_categories: 
        current_matches_to_filter = [job for job in current_matches_to_filter if job.get('category') in selected_categories]
    if selected_languages: 
        current_matches_to_filter = [job for job in current_matches_to_filter if any(lang in selected_languages for lang in get_job_languages_from_metadata(job.get('metadata', {})))]
    final_display_matches = [j for j in current_matches_to_filter if isinstance(j.get('score'), (int, float)) and j.get('score', 0) >= SIMILARITY_THRESHOLD]
    final_display_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
    final_display_matches = final_display_matches[:MAX_JOBS_TO_DISPLAY_PER_PAGE]

    tab_results, tab_feedback_analytics = st.tabs(["🎯 Matching Jobs", "📊 Feedback Analytics"])
    with tab_results:
        if not uploaded_file: st.info("👋 Upload your CV using the sidebar to find matching jobs!")
        elif not st.session_state.all_job_matches_cache and uploaded_file: st.warning("Initial job search yielded no results or failed.", icon="🤷")
        if final_display_matches:
            st.success(f"Displaying top {len(final_display_matches)} matches (Score ≥ {SIMILARITY_THRESHOLD:.0f}%).")
            for i, job_match in enumerate(final_display_matches):
                # Fix: Use 'job_id' instead of 'chroma_id' - this matches what ChromaDB returns
                job_unique_id = job_match.get('job_id', f"job_fallback_{i}")
                
                # Fix: Use correct field names that match ChromaDB metadata structure
                job_title = job_match.get('title', 'N/A')  # Changed from 'Title' to 'title'
                job_company = job_match.get('company', 'N/A')  # Changed from 'Company' to 'company'
                job_area_display = job_match.get('area', 'N/A')  # Changed from 'Area' to 'area'
                job_category_display = job_match.get('category', 'N/A')  # Changed from 'Category' to 'category'
                
                # Fix: Handle status from metadata properly
                job_status = job_match.get('metadata', {}).get('Status', 'unknown').capitalize()
                
                job_url = job_match.get('url', '#')
                job_score = job_match.get('score', 0.0)
                contributing_skills = job_match.get('contributing_skills', [])
                
                # Fix: Use correct field name for document content
                job_description_text = job_match.get('document', '')  # Changed from 'document_text' to 'document'
                
                # Use metadata for language detection
                job_languages_display = get_job_languages_from_metadata(job_match.get('metadata', {}))
                job_feedback_stats = feedback_aggregates["per_job"].get(job_unique_id, {"up": 0, "down": 0})
                with st.container(border=True):
                    main_cols = st.columns([5, 2])
                    with main_cols[0]:
                        st.markdown(f"#### {i+1}. {job_title}")
                        st.caption(f"**🏢 Company:** {job_company} | **📍 Location:** {job_area_display} | **🏷️ Category:** {job_category_display} | **🚦 Status:** `{job_status}`")
                        if job_languages_display: st.caption(f"**🗣️ Languages:** {', '.join(job_languages_display)}")
                        with st.expander("🔬 See Matching Skills Analysis", expanded=False):
                            if contributing_skills:
                                st.markdown("**Key CV skills contributing to this match:**")
                                for skill_text, skill_sim_score in contributing_skills: st.markdown(f"- `{skill_text}` (Contribution: {skill_sim_score:.2f})")
                            else: st.caption("Skill contribution analysis not available.")
                        action_cols = st.columns([1, 1])
                        with action_cols[0]:
                            if job_url and job_url != '#': st.link_button("Apply Now 🚀", url=job_url, type="primary", use_container_width=True)
                            else: st.button("Apply Now", disabled=True, use_container_width=True, help="Application link unavailable.")
                        with action_cols[1]:
                            cl_button_key = f"cl_btn_{job_unique_id}"
                            disable_cl = not (cover_letter_gen and job_description_text and st.session_state.get('cv_text_cache'))
                            cl_help_text = "Job/CV text missing." if not (job_description_text and st.session_state.get('cv_text_cache')) else "OpenAI API key missing." if not cover_letter_gen else "Draft cover letter"
                            if st.button("Draft Cover Letter 📄", key=cl_button_key, use_container_width=True, disabled=disable_cl, help=cl_help_text):
                                with st.spinner("🖋️ Drafting cover letter..."):
                                    gen_letter = cover_letter_gen.generate_cover_letter(job_description_text, st.session_state.cv_text_cache)
                                st.session_state.generated_cover_letters[job_unique_id] = gen_letter if (gen_letter and not gen_letter.lower().startswith("error:")) else "Error: Could not generate cover letter."
                        if job_unique_id in st.session_state.generated_cover_letters:
                            letter_content = st.session_state.generated_cover_letters[job_unique_id]
                            if not letter_content.lower().startswith("error:"):
                                st.markdown("**Generated Cover Letter Draft:**")
                                st.text_area("", value=letter_content, height=350, key=f"cl_txt_{job_unique_id}", help="AI draft. Review & edit carefully.")
                                pdf_bytes = create_pdf_from_text(letter_content)
                                if pdf_bytes: st.download_button("Download PDF 💾", data=pdf_bytes, file_name=f"CoverLetter_{re.sub(r'[^a-zA-Z0-9_]', '', job_title)[:30]}.pdf", mime="application/pdf", key=f"dl_pdf_{job_unique_id}")
                            else: st.error(letter_content, icon="⚠️")
                    with main_cols[1]:
                        st.metric("Match Score", f"{job_score:.1f}%", help="How well your CV skills match this job's profile.")
                        st.markdown("**Rate this match:**")
                        fb_key_suffix = f"fb_{job_unique_id}_{st.session_state.cv_upload_time or ''}"
                        current_rating = st.session_state.feedback_given_jobs.get(job_unique_id)
                        fb_cols = st.columns(2)
                        #cv_skill_count = len(st.session_state.cv_skills or []) # Not used if not writing to CSV
                        with fb_cols[0]:
                            if st.button("👍", key=f"up_{fb_key_suffix}", disabled=(current_rating is not None), use_container_width=True, type="primary" if current_rating=="up" else "secondary", help="Good match!"):
                                # Pass values for predicted_score and rank_displayed if available, otherwise None or empty strings
                                if record_feedback_local(st.session_state.session_id, st.session_state.cv_upload_time, job_unique_id, "up", 
                                                         predicted_score_val=job_score, rank_displayed_val=str(i+1)):
                                    st.session_state.feedback_given_jobs[job_unique_id] = "up"; st.rerun()
                        with fb_cols[1]:
                            if st.button("👎", key=f"down_{fb_key_suffix}", disabled=(current_rating is not None), use_container_width=True, type="primary" if current_rating=="down" else "secondary", help="Not a good match."):
                                if record_feedback_local(st.session_state.session_id, st.session_state.cv_upload_time, job_unique_id, "down", 
                                                         predicted_score_val=job_score, rank_displayed_val=str(i+1)):
                                    st.session_state.feedback_given_jobs[job_unique_id] = "down"; st.rerun()
                        st.caption(f"Community Votes: 👍{job_feedback_stats.get('up', 0)} | 👎{job_feedback_stats.get('down', 0)}")
                        if current_rating: st.success(f"You rated: {'👍' if current_rating == 'up' else '👎'}")
        elif st.session_state.all_job_matches_cache is not None and not final_display_matches and (selected_locations or selected_categories or selected_languages):
            st.info("No jobs match your current filter criteria.", icon="🧐")
        elif st.session_state.all_job_matches_cache is not None and not final_display_matches and uploaded_file:
            st.info(f"Found {len(st.session_state.all_job_matches_cache)} potential matches, but none scored {SIMILARITY_THRESHOLD:.0f}% or higher.", icon="📉")

    with tab_feedback_analytics:
        st.header("📈 Community Feedback Analytics")
        st.markdown("Insights from all user feedback (saved locally). Your ratings help improve this overview!")
        st.divider()
        total_up = feedback_aggregates.get("total_up", 0); total_down = feedback_aggregates.get("total_down", 0)
        total_votes = total_up + total_down
        if total_votes > 0:
            st.markdown("**Key Metrics**"); satisfaction_score = (total_up / total_votes) * 100 if total_votes > 0 else 0
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Total Feedback Votes", total_votes); m_col2.metric("Overall Satisfaction", f"{satisfaction_score:.1f}%")
            st.divider(); st.markdown("**Feedback Distribution**")
            pie_data = pd.DataFrame({'Rating Type': ['Good Matches 👍', 'Bad Matches 👎'], 'Votes': [total_up, total_down]})
            try:
                fig_pie = px.pie(pie_data, values='Votes', names='Rating Type', color='Rating Type', color_discrete_map={'Good Matches 👍':'#2ECC71', 'Bad Matches 👎':'#E74C3C'}, hole=0.3)
                fig_pie.update_layout(legend_title_text='Feedback', margin=dict(t=20,b=20,l=0,r=0)); fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            except Exception as plot_err: st.error(f"Pie chart error: {plot_err}")
            st.divider(); st.markdown("**Feedback Trend Over Time (Daily)**")
            if (feedback_df is not None and not feedback_df.empty and 'ts' in feedback_df.columns and pd.api.types.is_datetime64_any_dtype(feedback_df['ts'])):
                try:
                    daily_fb = feedback_df.set_index('ts').resample('D')['rt'].value_counts().unstack(fill_value=0)
                    if 'up' not in daily_fb.columns: daily_fb['up'] = 0
                    if 'down' not in daily_fb.columns: daily_fb['down'] = 0
                    daily_fb = daily_fb.rename(columns={'up': 'Good 👍', 'down': 'Bad 👎'})
                    if not daily_fb.empty and (daily_fb['Good 👍'].sum() > 0 or daily_fb['Bad 👎'].sum() > 0):
                        fig_time = px.area(daily_fb, y=['Good 👍', 'Bad 👎'], labels={"ts": "Date", "value": "Ratings", "variable": "Rating"}, color_discrete_map={'Good 👍': '#2ECC71', 'Bad 👎': '#E74C3C'}, markers=True)
                        fig_time.update_layout(hovermode="x unified", legend_title_text='Rating', yaxis_title="Number of Ratings")
                        st.plotly_chart(fig_time, use_container_width=True)
                    else: st.info("Not enough data for trend.", icon="📈")
                except Exception as time_plot_err: st.error(f"Trend chart error: {time_plot_err}")
            else: st.info("Not enough timestamp data for trend.", icon="⏳")
        else: st.info("📊 No feedback data to display analytics.", icon="✏️")

# Footer
st.markdown("---")
st.caption("CV Job Matcher | An MLOps Project | For Educational Purposes")
