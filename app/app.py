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
    page_icon="🎯", # Changed icon
    layout="wide",
    initial_sidebar_state="expanded" # Keep sidebar open initially
)

# --- Constants and Normalization Data ---
MIN_CV_LENGTH_CHARS = 150
MAX_CV_LENGTH_CHARS = 20000
LOCAL_FEEDBACK_FILENAME = "feedback_log.csv" 
SIMILARITY_THRESHOLD = 40.0 
MAX_JOBS_TO_DISPLAY_PER_PAGE = 5 # For pagination later if needed, for now, it's the total max
TOP_N_RESULTS_FROM_SEARCH = int(os.getenv('TOP_N_RESULTS_FOR_APP_QUERY', '75')) # Fetch more for better filtering

CANONICAL_LANGUAGES_FOR_FILTER = ["English", "Danish", "German", "Spanish", "French", "Norwegian", "Swedish"]
CANONICAL_AREAS_FOR_FILTER = sorted([
    "Copenhagen", "Aarhus", "Odense", "Aalborg", "Esbjerg", "Randers", "Kolding", "Horsens", "Vejle", "Roskilde",
    "Herning", "Hørsholm", "Silkeborg", "Næstved", "Fredericia", "Viborg", "Køge", "Holstebro", "Taastrup", "Slagelse",
    "Hillerød", "Sønderborg", "Svendborg", "Hjørning", "Holbæk", "Frederikshavn", "Nørresundby", "Ringsted", "Haderslev",
    "Skive", "Ølstykke-Stenløse", "Nykøbing Falster", "Greve Strand", "Kalundborg", "Ballerup", "Rødovre", "Lyngby",
    "Albertslund", "Hvidovre", "Glostrup", "Ishøj", "Birkerød", "Farum", "Frederikssund", "Brøndby Strand",
    "Skanderborg", "Hedensted", "Frederiksværk", "Lillerød", "Solrød Strand", "Other/Unmapped Area"
])

# --- Normalization Functions (Adapted for App Context) ---
def app_normalize_area(raw_area_str):
    if not isinstance(raw_area_str, str) or not raw_area_str.strip():
        return "Other/Unmapped Area"
    
    normalized_area = raw_area_str.strip()
    for canonical_area_val in CANONICAL_AREAS_FOR_FILTER:
        if normalized_area.lower() == canonical_area_val.lower():
            return canonical_area_val
        if canonical_area_val.lower() in normalized_area.lower() and canonical_area_val != "Other/Unmapped Area":
            return canonical_area_val 
            
    city_name_mapping = {'København': 'Copenhagen', 'Århus': 'Aarhus', 'Storkøbenhavn': 'Copenhagen', 'Hovedstadsområdet': 'Copenhagen'}
    for danish_key, canonical_value in city_name_mapping.items():
        if danish_key.lower() in normalized_area.lower():
            return canonical_value
    return "Other/Unmapped Area" 

def get_job_languages_from_metadata(metadata_dict):
    extracted_languages_set = set()
    lang_req_json_str = metadata_dict.get("Language_Requirements_Json_Str")
    if lang_req_json_str:
        try:
            lang_list_of_dicts = json.loads(lang_req_json_str)
            if isinstance(lang_list_of_dicts, list):
                for lang_entry in lang_list_of_dicts:
                    if isinstance(lang_entry, dict):
                        lang_name = lang_entry.get("language")
                        if isinstance(lang_name, str) and lang_name.strip():
                            for canonical in CANONICAL_LANGUAGES_FOR_FILTER:
                                if canonical.lower() == lang_name.strip().lower():
                                    extracted_languages_set.add(canonical); break
        except: pass # Silently ignore parsing errors for this field for now

    for key, value in metadata_dict.items():
        if key.startswith("lang_") and key.endswith("_proficiency"):
            lang_name_from_key = key.replace("lang_", "").replace("_proficiency", "")
            for canonical in CANONICAL_LANGUAGES_FOR_FILTER:
                if canonical.lower() == lang_name_from_key.lower():
                    extracted_languages_set.add(canonical); break
                    
    detected_ad_lang_raw = metadata_dict.get("Detected_Ad_Language")
    if isinstance(detected_ad_lang_raw, str) and detected_ad_lang_raw.strip() and detected_ad_lang_raw != "Unknown":
        for canonical in CANONICAL_LANGUAGES_FOR_FILTER:
            if canonical.lower() == detected_ad_lang_raw.strip().lower():
                extracted_languages_set.add(canonical); break
                
    return sorted(list(extracted_languages_set))

# Initialize Cover Letter Generator
@st.cache_resource
def get_cover_letter_generator():
    try: return CoverLetterGenerator() 
    except Exception as e: st.error(f"Cover Letter Generator Error: {e}. Check OPENAI_API key."); return None
cover_letter_gen = get_cover_letter_generator()

# PDF Generation Function
class PDF(FPDF):
    def header(self): pass 
    def footer(self): pass 

def create_pdf_from_text(text_content):
    try:
        pdf = PDF(); pdf.add_page()
        font_path = "DejaVuSans.ttf" 
        font_name = "DejaVu"
        try:
            if os.path.exists(font_path): pdf.add_font(font_name, "", font_path, uni=True)
            else: raise RuntimeError(f"Font file {font_path} not found.")
            pdf.set_font(font_name, size=11)
        except RuntimeError:
            print(f"Warning: Custom font not found. Falling back to Arial.")
            try: pdf.set_font("Arial", size=11)
            except RuntimeError: pdf.set_font("Helvetica", size=11) # Further fallback
        
        pdf.write(5, text_content) 
        pdf_output_bytes = pdf.output(dest='S')
        if not pdf_output_bytes: st.error("PDF generation resulted in empty output."); return None
        return pdf_output_bytes
    except Exception as e: st.error(f"Error generating PDF: {e}"); print(f"PDF error: {traceback.format_exc()}"); return None

# File Reading and Feedback functions (no major changes needed, keeping them concise here for brevity)
def read_cv_file(uploaded_file): # Condensed
    if not uploaded_file: return None
    try:
        file_name=uploaded_file.name; file_ext=os.path.splitext(file_name)[1].lower(); cv_text=""
        with st.spinner(f"Reading `{file_name}`..."):
            content_bytes = uploaded_file.getvalue()
            if file_ext=='.pdf':
                r=PyPDF2.PdfReader(io.BytesIO(content_bytes)); cv_text="".join([(p.extract_text() or "") for p in r.pages])
            elif file_ext=='.docx':
                d=docx.Document(io.BytesIO(content_bytes)); cv_text="\n".join([p.text for p in d.paragraphs])
            elif file_ext=='.md':
                h=markdown.markdown(content_bytes.decode("utf-8",errors="ignore")); cv_text=re.sub('<[^>]*>',' ',h).strip()
            elif file_ext=='.txt':
                cv_text=content_bytes.decode("utf-8",errors="ignore")
            else: st.error(f"Unsupported: `{file_ext}`."); return None
        final_text=cv_text.strip()
        if not final_text: st.warning("No text extracted."); return None
        return final_text
    except Exception as e: st.error(f"Error reading '{uploaded_file.name}'."); print(f"Read error: {traceback.format_exc()}"); return None

def initialize_local_feedback_csv(): # Condensed
    if not os.path.exists(LOCAL_FEEDBACK_FILENAME):
        try:
            with open(LOCAL_FEEDBACK_FILENAME,'w',newline='',encoding='utf-8') as f: csv.writer(f).writerow(["ts","sid","cv_ts","jid","rt","cv_sc","jt"])
        except Exception as e: st.error(f"Feedback file init error: {e}"); return False
    return True

def record_feedback_local(sid, cv_ts, jid, rt, cv_sc, jt): # Condensed
    if not initialize_local_feedback_csv(): return False
    try:
        with open(LOCAL_FEEDBACK_FILENAME,'a',newline='',encoding='utf-8') as f: csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(),sid,cv_ts,jid,rt,cv_sc,jt])
        st.toast("Feedback saved!", icon="👍"); return True
    except Exception as e: st.error(f"Feedback record error: {e}"); return False

def load_and_process_feedback(): # Condensed
    def_res = {"aggregates":{"per_job":{},"total_up":0,"total_down":0},"dataframe":pd.DataFrame(columns=["ts","sid","cv_ts","jid","rt","cv_sc","jt"])}
    if not initialize_local_feedback_csv() or not os.path.isfile(LOCAL_FEEDBACK_FILENAME): return def_res
    try:
        df=pd.read_csv(LOCAL_FEEDBACK_FILENAME,names=["ts","sid","cv_ts","jid","rt","cv_sc","jt"],header=0,low_memory=False) # Added names and header=0
        if df.empty: return def_res
        req_c=['rt','jid','ts']; 
        if not all(c in df.columns for c in req_c):print("FB CSV missing cols"); return def_res
        tu=int(df['rt'].value_counts().get('up',0)); td=int(df['rt'].value_counts().get('down',0))
        pj={};
        try:
            jc=df.groupby('jid')['rt'].value_counts().unstack(fill_value=0)
            if 'up' not in jc.columns: jc['up']=0
            if 'down' not in jc.columns: jc['down']=0
            pj=jc.apply(lambda r:{"up":int(r['up']),"down":int(r['down'])},axis=1).to_dict()
        except: pass
        aggs={"per_job":pj,"total_up":tu,"total_down":td}
        df['ts']=pd.to_datetime(df['ts'],errors='coerce');df.dropna(subset=['ts'],inplace=True)
        return {"aggregates":aggs,"dataframe":df}
    except: print(f"FB process error: {traceback.format_exc()}"); return def_res


# --- Streamlit App State Initialization ---
if 'session_id' not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
if 'cv_upload_time' not in st.session_state: st.session_state.cv_upload_time = None
if 'feedback_given_jobs' not in st.session_state: st.session_state.feedback_given_jobs = {} 
if 'cv_skills' not in st.session_state: st.session_state.cv_skills = None
if 'generated_cover_letters' not in st.session_state: st.session_state.generated_cover_letters = {}
if 'all_job_matches_cache' not in st.session_state: st.session_state.all_job_matches_cache = None
if 'cv_text_cache' not in st.session_state: st.session_state.cv_text_cache = None


# --- App Header & Intro ---
st.title("👨‍💼🇩🇰 CV Job Matcher Pro") 
st.subheader("Unlock Your Next Career Move in Denmark!")
st.markdown("Upload your CV, and our AI will find jobs that truly match your skills, help you understand why, and even draft a cover letter.")
st.markdown("---")

# --- Prerequisite Checks ---
if not all([os.getenv('EMBEDDING_API_URL'), os.getenv('CHROMA_HOST'), os.getenv('CHROMA_PORT'), os.getenv('CHROMA_COLLECTION')]):
    st.error("Backend services are not fully configured. Please check `.env` settings. The app cannot function without them.")
    st.stop()
if not os.getenv("OPENAI_API"):
    st.warning("OpenAI API key not found. Cover letter generation will be disabled.", icon="🔒")
    cover_letter_gen = None 

# --- Main App Layout ---
# Sidebar for Upload and Filters
with st.sidebar:
    st.header("🚀 Get Started")
    uploaded_file = st.file_uploader("1. Upload Your CV", type=['pdf', 'docx', 'txt', 'md'], 
                                    key="cv_uploader_key",
                                    help="Supports PDF, DOCX, Markdown, and TXT files.",
                                    on_change=lambda: st.session_state.update(all_job_matches_cache=None, generated_cover_letters={}, cv_text_cache=None, cv_skills=None))
    st.markdown("---")
    st.header("🔍 Filter Job Matches")
    selected_locations = st.multiselect("Job Locations (Area)", options=CANONICAL_AREAS_FOR_FILTER, placeholder="Any Location")
    selected_languages = st.multiselect("Required Languages", options=CANONICAL_LANGUAGES_FOR_FILTER, placeholder="Any Language")
    
    st.markdown("---")
    st.info("Tip: Upload your CV first, then apply filters to the results.")
    st.caption(f"Matching up to {TOP_N_RESULTS_FROM_SEARCH} jobs, displaying top {MAX_JOBS_TO_DISPLAY_PER_PAGE} after filters.")


# Main content area
placeholder_processing_status = st.empty() # For global status messages

# Load feedback data
feedback_result = load_and_process_feedback()
feedback_aggregates = feedback_result["aggregates"]
feedback_df = feedback_result["dataframe"]


if uploaded_file is not None:
    if st.session_state.all_job_matches_cache is None: # Process only on new upload or if cache is empty
        with placeholder_processing_status.container(): # Use container for status
            with st.spinner(f"Analyzing `{uploaded_file.name}`... This might take a few moments."):
                st.session_state.cv_upload_time = datetime.now(timezone.utc).isoformat()
                st.session_state.feedback_given_jobs = {} 
                st.session_state.generated_cover_letters = {}
                
                cv_text = read_cv_file(uploaded_file)
                st.session_state.cv_text_cache = cv_text # Cache raw CV text
                
                if cv_text:
                    # Skill Extraction
                    with st.spinner("🤖 Extracting skills from your CV..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue()); temp_cv_path = tmp_file.name
                        cv_skills_list = get_extracted_skills_from_file(temp_cv_path); os.unlink(temp_cv_path)
                    if not cv_skills_list: placeholder_processing_status.error("Could not extract skills. Try a different CV."); st.stop()
                    st.session_state.cv_skills = cv_skills_list
                    if len(cv_skills_list) < 50:
                        with st.expander("💡 View Extracted CV Skills", expanded=False): st.caption(f"{len(cv_skills_list)} skills: {', '.join(cv_skills_list)}")
                    
                    # Embedding
                    with st.spinner("🧬 Generating CV skills embedding..."):
                        cv_skill_embedding_vec = generate_embedding_for_skills(cv_skills_list)
                    if cv_skill_embedding_vec is None: placeholder_processing_status.error("Could not generate CV embedding."); st.stop()
                    
                    # Job Matching
                    with st.spinner('🧠 Searching for matching jobs...'):
                        all_matches_db, method_used = find_similar_jobs(
                            cv_skill_embedding=cv_skill_embedding_vec, cv_skills=cv_skills_list, 
                            top_n=TOP_N_RESULTS_FROM_SEARCH, active_only=True)
                        st.session_state.all_job_matches_cache = all_matches_db if all_matches_db is not None else []
                else: # cv_text is None
                    placeholder_processing_status.error("Could not read CV content."); st.session_state.all_job_matches_cache = []
            placeholder_processing_status.success(f"CV processing complete! Found {len(st.session_state.all_job_matches_cache)} potential matches.")
            time.sleep(2) # Keep success message for a bit
            placeholder_processing_status.empty()


# Apply Filters and Display Results
if st.session_state.all_job_matches_cache is not None:
    current_matches_to_filter = list(st.session_state.all_job_matches_cache) # Work with a copy
    
    # Apply Location Filter
    if selected_locations:
        filtered_by_location = []
        for job in current_matches_to_filter:
            job_area_for_filter = app_normalize_area(job.get('Area', ''))
            if job_area_for_filter in selected_locations:
                filtered_by_location.append(job)
        current_matches_to_filter = filtered_by_location

    # Apply Language Filter
    if selected_languages:
        filtered_by_language = []
        for job in current_matches_to_filter:
            job_langs = get_job_languages_from_metadata(job) 
            if any(lang in selected_languages for lang in job_langs):
                filtered_by_language.append(job)
        current_matches_to_filter = filtered_by_language
        
    final_display_matches = [j for j in current_matches_to_filter if isinstance(j.get('score'), (int, float)) and j.get('score', 0) >= SIMILARITY_THRESHOLD]
    final_display_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
    final_display_matches = final_display_matches[:MAX_JOBS_TO_DISPLAY_PER_PAGE]

    # Tabs for results and analytics
    tab_results, tab_feedback_analytics = st.tabs(["🎯 Matching Jobs", "📊 Feedback Analytics"])
    
    with tab_results:
        if not uploaded_file:
             st.info("👋 Upload your CV using the sidebar to find matching jobs!")
        elif not st.session_state.all_job_matches_cache and uploaded_file :
             st.warning("Initial job search yielded no results or failed. Try a different CV or check service status.", icon="🤷")

        if final_display_matches:
            st.success(f"Displaying top {len(final_display_matches)} matches based on your CV and filters (Score ≥ {SIMILARITY_THRESHOLD:.0f}%).")
            for i, job_match in enumerate(final_display_matches):
                job_unique_id = job_match.get('chroma_id', f"job_fallback_{i}")
                job_title = job_match.get('Title', 'N/A')
                job_company = job_match.get('Company', 'N/A')
                job_area_display = app_normalize_area(job_match.get('Area', ''))
                job_status = job_match.get('Status', 'unknown').capitalize()
                job_url = job_match.get('url', '#')
                job_score = job_match.get('score', 0.0)
                contributing_skills = job_match.get('contributing_skills', [])
                job_description_text = job_match.get('document_text', '') 
                job_languages_display = get_job_languages_from_metadata(job_match)

                job_feedback_stats = feedback_aggregates["per_job"].get(job_unique_id, {"up": 0, "down": 0})

                with st.container(border=True):
                    main_cols = st.columns([5, 2]) # Job Info | Score & Feedback
                    with main_cols[0]: # Job Info Column
                        st.markdown(f"#### {i+1}. {job_title}")
                        st.caption(f"**🏢 Company:** {job_company} | **📍 Location:** {job_area_display} | **🚦 Status:** `{job_status}`")
                        if job_languages_display:
                            st.caption(f"**🗣️ Languages:** {', '.join(job_languages_display)}")
                        
                        with st.expander("🔬 See Matching Skills Analysis", expanded=False):
                            if contributing_skills:
                                st.markdown("**Key CV skills contributing to this match:**")
                                for skill_text, skill_sim_score in contributing_skills: 
                                    st.markdown(f"- `{skill_text}` (Contribution: {skill_sim_score:.2f})")
                            else: st.caption("Skill contribution analysis not available or no strong contributing skills found.")
                        
                        action_cols = st.columns([1,1]) 
                        with action_cols[0]: 
                            if job_url and job_url != '#': st.link_button("Apply Now 🚀", url=job_url, type="primary", use_container_width=True)
                            else: st.button("Apply Now", disabled=True, use_container_width=True, help="Application link unavailable.")
                        with action_cols[1]: 
                            cl_button_key = f"cl_btn_{job_unique_id}"
                            disable_cl = not (cover_letter_gen and job_description_text and st.session_state.get('cv_text_cache'))
                            cl_help_text = "Job description or CV text missing." if not (job_description_text and st.session_state.get('cv_text_cache')) else "OpenAI API key missing." if not cover_letter_gen else "Draft a cover letter"
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
                    
                    with main_cols[1]: # Score & Feedback Column
                        st.metric("Match Score", f"{job_score:.1f}%", help="How well your CV skills match this job's profile.")
                        st.markdown("**Rate this match:**")
                        fb_key_suffix = f"fb_{job_unique_id}_{st.session_state.cv_upload_time or ''}"
                        current_rating = st.session_state.feedback_given_jobs.get(job_unique_id)
                        fb_cols = st.columns(2)
                        cv_skill_count = len(st.session_state.cv_skills or [])
                        with fb_cols[0]:
                            if st.button("👍", key=f"up_{fb_key_suffix}", disabled=(current_rating is not None), use_container_width=True, type="primary" if current_rating=="up" else "secondary", help="Good match!"):
                                if record_feedback_local(st.session_state.session_id, st.session_state.cv_upload_time, job_unique_id, "up", cv_skill_count, job_title):
                                    st.session_state.feedback_given_jobs[job_unique_id] = "up"; st.rerun()
                        with fb_cols[1]:
                            if st.button("👎", key=f"down_{fb_key_suffix}", disabled=(current_rating is not None), use_container_width=True, type="primary" if current_rating=="down" else "secondary", help="Not a good match."):
                                if record_feedback_local(st.session_state.session_id, st.session_state.cv_upload_time, job_unique_id, "down", cv_skill_count, job_title):
                                    st.session_state.feedback_given_jobs[job_unique_id] = "down"; st.rerun()
                        st.caption(f"Community Votes: 👍{job_feedback_stats.get('up', 0)} | 👎{job_feedback_stats.get('down', 0)}")
                        if current_rating: st.success(f"You rated: {'👍' if current_rating == 'up' else '👎'}")
            
        elif st.session_state.all_job_matches_cache is not None and not final_display_matches and (selected_locations or selected_languages):
            st.info("No jobs match your current filter criteria. Try adjusting the filters or uploading a different CV.", icon="🧐")
        elif st.session_state.all_job_matches_cache is not None and not final_display_matches and uploaded_file:
            st.info(f"Found {len(st.session_state.all_job_matches_cache)} potential matches, but none scored {SIMILARITY_THRESHOLD:.0f}% or higher. Try refining your CV or broadening your search.", icon="📉")


        with tab_feedback_analytics: 
            st.header("📈 Community Feedback Analytics")
            st.markdown("Insights from all user feedback (saved locally). Your ratings help improve this overview!")
            st.divider()
            # ... (Feedback analytics display logic - remains largely the same as your previous version) ...
            total_up = feedback_aggregates.get("total_up", 0); total_down = feedback_aggregates.get("total_down", 0)
            total_votes = total_up + total_down
            if total_votes > 0:
                st.markdown("**Key Metrics**"); satisfaction_score = (total_up / total_votes) * 100 if total_votes > 0 else 0
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Total Feedback Votes", total_votes, help="Total 👍 & 👎 ratings."); 
                m_col2.metric("Overall Satisfaction", f"{satisfaction_score:.1f}%", help="% of 👍 ratings.")
                st.divider()
                st.markdown("**Feedback Distribution**")
                pie_data = pd.DataFrame({'Rating Type': ['Good Matches 👍', 'Bad Matches 👎'], 'Votes': [total_up, total_down]})
                try:
                    fig_pie = px.pie(pie_data, values='Votes', names='Rating Type', color='Rating Type', color_discrete_map={'Good Matches 👍':'#2ECC71', 'Bad Matches 👎':'#E74C3C'}, hole=0.3)
                    fig_pie.update_layout(legend_title_text='Feedback', margin=dict(t=20,b=20,l=0,r=0)); fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                except Exception as plot_err: st.error(f"Pie chart error: {plot_err}")
                st.divider()
                st.markdown("**Feedback Trend Over Time (Daily)**")
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
            else: st.info("📊 No feedback yet. Rate matches to see community analytics!", icon="✏️")


# Footer
st.markdown("---")
st.caption("CV Job Matcher Pro | An MLOps Project | For Educational Purposes")
