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

# Load environment variables for local dev
load_dotenv()

# Import matching functions
try:
    from cv_match import find_similar_jobs
except ImportError as import_err:
    st.error(f"**Initialization Error:** Could not import matching functions from 'cv_match.py': {import_err}")
    st.info("Please ensure 'cv_match.py' is in the same directory as 'app.py'.")
    st.stop()
except Exception as general_import_err:
    st.error(f"**Initialization Error:** Unexpected error importing 'cv_match': {general_import_err}")
    st.stop()

# Hugging Face Hub integration
try:
    from huggingface_hub import hf_hub_download, upload_file
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    st.error("**Initialization Error:** Could not import `huggingface_hub`. Please add it to requirements.txt.")
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
REMOTE_FEEDBACK_FILENAME = "feedback_log.csv"
SIMILARITY_THRESHOLD = 40.0
MAX_JOBS_TO_DISPLAY = 5

# Hugging Face configuration
HF_DATASET_REPO_ID = "Lpiziks2/cv-matcher-feedback"
HF_TOKEN = os.getenv("HF_TOKEN")
PLACEHOLDER_REPO_ID = "YOUR_USERNAME/YOUR_DATASET_REPO_NAME"

# Function to read CV file in various formats
def read_cv_file(uploaded_file):
    if not uploaded_file:
        return None
    
    try:
        file_name = uploaded_file.name
        file_ext = os.path.splitext(file_name)[1].lower()
        cv_text = ""
        
        with st.status(f"Reading `{file_name}`...", expanded=False) as status:
            if file_ext == '.pdf':
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                if not pdf_reader.pages:
                    status.update(label="Warning: PDF empty or unreadable.", state="warning")
                    return None
                    
                for page in pdf_reader.pages:
                    cv_text += (page.extract_text() or "") + "\n"
                    
                if not cv_text.strip():
                    status.update(label="Warning: No text extracted from PDF.", state="warning")
                    
            elif file_ext == '.docx':
                doc = docx.Document(uploaded_file)
                cv_text = "\n".join([p.text for p in doc.paragraphs if p.text])
                
                if not cv_text.strip():
                    status.update(label="Warning: No text found in DOCX.", state="warning")
                    
            elif file_ext == '.md':
                md_bytes = uploaded_file.read()
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
                txt_bytes = uploaded_file.read()
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

@st.cache_resource
def sync_feedback_csv():
    """Downloads or creates feedback CSV file"""
    local_path = os.path.join(".", LOCAL_FEEDBACK_FILENAME)

    # Handle missing token case
    if not HF_TOKEN:
        print("WARN: HF_TOKEN missing. Using temporary local file if needed.")
        if not os.path.exists(local_path):
            try:
                with open(local_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "session_id", "cv_upload_time", "job_chroma_id", "rating"])
            except Exception as e:
                print(f"Error creating temporary local feedback file: {e}")
                return None
        return local_path

    # Try downloading from Hub
    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_DATASET_REPO_ID,
            filename=REMOTE_FEEDBACK_FILENAME,
            repo_type="dataset",
            token=HF_TOKEN,
            cache_dir="./hf_cache",
            force_filename=LOCAL_FEEDBACK_FILENAME
        )
        print(f"Downloaded/found feedback CSV at: {downloaded_path}")
        
        if downloaded_path != local_path:
            import shutil
            shutil.copy2(downloaded_path, local_path)
            print(f"Copied synced file from {downloaded_path} to {local_path}")
            
        return local_path

    # Handle 404 case (file doesn't exist yet)
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            print(f"Feedback file not found on Hub. Creating default.")
            if not os.path.exists(local_path):
                try:
                    with open(local_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(["timestamp", "session_id", "cv_upload_time", "job_chroma_id", "rating"])
                except Exception as creation_err:
                    print(f"Error creating local feedback file: {creation_err}")
                    st.error("Failed to initialize local feedback file.")
                    return None
            return local_path
        else:
            st.error(f"Error downloading feedback CSV (HTTP {e.response.status_code}): {e}", icon="☁️")
            print(f"Error downloading feedback CSV: {e}")
            
            # Create local fallback if needed
            if not os.path.exists(local_path):
                try:
                    with open(local_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(["timestamp", "session_id", "cv_upload_time", "job_chroma_id", "rating"])
                except Exception as fallback_err:
                    print(f"Error creating fallback file: {fallback_err}")
                    return None
            return local_path

    # Handle other unexpected errors
    except Exception as e:
        st.error(f"Unexpected error during feedback sync: {e}", icon="⚙️")
        print(f"Sync error: {traceback.format_exc()}")
        
        # Create local fallback if needed
        if not os.path.exists(local_path):
            try:
                with open(local_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "session_id", "cv_upload_time", "job_chroma_id", "rating"])
            except Exception as final_err:
                print(f"Error creating final fallback file: {final_err}")
                return None
        return local_path

def record_feedback_hub(session_id, cv_upload_time, job_chroma_id, rating):
    """Records user feedback and uploads to Hugging Face Hub"""
    if not HF_TOKEN:
        st.warning("Feedback can't be saved: Hugging Face Token not configured.", icon="🔒")
        print("WARN: HF_TOKEN missing, skipping feedback persistence.")
        return False

    local_csv_path = sync_feedback_csv()
    if not local_csv_path or not os.path.exists(local_csv_path):
        st.error("Could not access or create local feedback file.", icon="❌")
        return False

    try:
        # Add feedback to local file
        timestamp = datetime.now(timezone.utc).isoformat()
        new_row = [timestamp, session_id, cv_upload_time, job_chroma_id, rating]
        needs_header = os.path.getsize(local_csv_path) == 0
        
        with open(local_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if needs_header:
                print("WARN: Local feedback file was empty, adding header.")
                writer.writerow(["timestamp", "session_id", "cv_upload_time", "job_chroma_id", "rating"])
            writer.writerow(new_row)
            
        print(f"Appended feedback locally to {local_csv_path}")

        # Upload to Hub
        with st.spinner("💾 Saving feedback..."):
            try:
                upload_file(
                    path_or_fileobj=local_csv_path,
                    path_in_repo=REMOTE_FEEDBACK_FILENAME,
                    repo_id=HF_DATASET_REPO_ID,
                    repo_type="dataset",
                    token=HF_TOKEN,
                    commit_message=f"Append feedback: {rating} for job {job_chroma_id}",
                    commit_description=f"Feedback recorded via Streamlit app. Session: {session_id}"
                )
                print(f"Uploaded updated feedback to Hub repo {HF_DATASET_REPO_ID}")
                st.toast("Feedback saved!", icon="✅")
                sync_feedback_csv.clear()  # Clear cache on success
                return True
                
            except Exception as upload_err:
                st.error(f"Error saving feedback to Hub: {upload_err}", icon="☁️")
                print(f"Error uploading feedback: {traceback.format_exc()}")
                return False

    except Exception as e:
        st.error(f"Error recording feedback locally: {e}", icon="❌")
        print(f"Error recording feedback: {traceback.format_exc()}")
        return False

def load_and_process_feedback():
    """Process feedback data from CSV for analytics"""
    print("Loading and processing feedback...")
    local_csv_path = sync_feedback_csv()
    
    default_result = {
        "aggregates": {"per_job": {}, "total_up": 0, "total_down": 0},
        "dataframe": pd.DataFrame(columns=["timestamp", "session_id", "cv_upload_time", "job_chroma_id", "rating"])
    }

    if not local_csv_path or not os.path.isfile(local_csv_path):
        print(f"Feedback file path invalid or missing")
        return default_result

    try:
        df = pd.read_csv(local_csv_path, low_memory=False)
        per_job_feedback = {}
        total_up = 0
        total_down = 0

        if df.empty:
            print("Feedback CSV is empty")
            aggregates = {"per_job": per_job_feedback, "total_up": total_up, "total_down": total_down}
            if not all(col in df.columns for col in default_result["dataframe"].columns):
                df = pd.DataFrame(columns=default_result["dataframe"].columns)
            return {"aggregates": aggregates, "dataframe": df}

        # Check for required columns
        required_cols = ['rating', 'job_chroma_id', 'timestamp']
        if not all(col in df.columns for col in required_cols):
            print(f"Feedback CSV missing required columns")
            st.warning("Feedback file format incorrect. Stats might be incomplete.", icon="⚠️")
            aggregates = {"per_job": {}, "total_up": 0, "total_down": 0}
            valid_cols = [col for col in default_result["dataframe"].columns if col in df.columns]
            return {"aggregates": aggregates, "dataframe": df[valid_cols] if valid_cols else default_result["dataframe"]}

        # Calculate totals
        total_counts = df['rating'].value_counts()
        total_up = int(total_counts.get('up', 0))
        total_down = int(total_counts.get('down', 0))

        # Calculate per-job stats
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

        # Process timestamps for trend analysis
        try:
            df_processed = df.copy()
            df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], errors='coerce')
            original_rows = len(df_processed)
            df_processed.dropna(subset=['timestamp'], inplace=True)
            dropped_rows = original_rows - len(df_processed)
            if dropped_rows > 0:
                print(f"Dropped {dropped_rows} rows with invalid timestamps")
        except Exception as time_e:
            print(f"Error parsing timestamps: {time_e}")
            st.warning("Could not process timestamps for trend analysis.", icon="⚠️")
            return {"aggregates": aggregates, "dataframe": df}

        print(f"Loaded feedback: Up={total_up}, Down={total_down}, Rows={len(df_processed)}")
        return {"aggregates": aggregates, "dataframe": df_processed}

    except pd.errors.EmptyDataError:
        print("Feedback file is empty")
        return default_result
    except Exception as e:
        print(f"Error processing feedback: {traceback.format_exc()}")
        st.error(f"Could not process feedback file. Stats may be unavailable.", icon="❌")
        return default_result

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'cv_upload_time' not in st.session_state:
    st.session_state.cv_upload_time = None
if 'feedback_given_jobs' not in st.session_state:
    st.session_state.feedback_given_jobs = set()

# App Header & Intro
st.title("👨‍💼🇩🇰 CV Job Matcher")
st.markdown("Unlock your next career move in Denmark. Upload your CV and let our AI find jobs that **truly match your skills.**")
st.markdown("---")

st.markdown(f"""
**How It Works**
1.  You **upload your CV** (PDF, DOCX, TXT, MD are cool).
2.  We **analyze the content** – understanding what you actually *do*.
3.  The system **compares your profile** to current, active job postings across Denmark.
4.  You get the **top results**, ranked by how well they match (score ≥ {SIMILARITY_THRESHOLD:.0f}%).
5.  Your feedback (👍/👎) is **saved securely** to help improve future matches! (Using Hugging Face Datasets)

Ready to dive in?
""")
st.markdown("---")

# Configuration & Prerequisite Checks
if HF_DATASET_REPO_ID == PLACEHOLDER_REPO_ID:
    st.error(f"Config Error: HF_DATASET_REPO_ID looks like a placeholder. Please update the code.")
    st.stop()
if not HF_TOKEN:
    st.warning("Warning: Hugging Face Token not found. Feedback won't be saved persistently.", icon="🔒")

# Check for backend settings
embedding_api_url = os.getenv('EMBEDDING_API_URL')
chroma_host = os.getenv('CHROMA_HOST')
chroma_port = os.getenv('CHROMA_PORT')
chroma_collection = os.getenv('CHROMA_COLLECTION')

if not all([embedding_api_url, chroma_host, chroma_port, chroma_collection]):
    missing_configs = [name for name, var in [
        ('EMBEDDING_API_URL', embedding_api_url),
        ('CHROMA_HOST', chroma_host),
        ('CHROMA_PORT', chroma_port),
        ('CHROMA_COLLECTION', chroma_collection)
    ] if not var]
    st.error(f"Missing backend settings: {', '.join(missing_configs)}. Check .env or Space secrets.")
    st.stop()

# File Upload
st.subheader("Upload Your CV 🚀")
uploaded_file = st.file_uploader("Choose CV file", type=['pdf', 'docx', 'txt', 'md'], 
                                label_visibility="collapsed", key="cv_uploader")

# Load feedback data
feedback_result = load_and_process_feedback()
feedback_aggregates = feedback_result["aggregates"]
feedback_df = feedback_result["dataframe"]

# Main Processing Logic
if uploaded_file is not None:
    # CV Processing
    file_process_placeholder = st.empty()
    file_process_placeholder.info(f"Processing `{uploaded_file.name}`...")
    st.session_state.cv_upload_time = datetime.now(timezone.utc).isoformat()
    st.session_state.feedback_given_jobs = set()
    cv_text = read_cv_file(uploaded_file)
    file_process_placeholder.empty()

    if cv_text:
        text_length = len(cv_text)
        if not (MIN_CV_LENGTH_CHARS <= text_length <= MAX_CV_LENGTH_CHARS):
            st.warning(f"CV Text Length: {text_length}. Recommended: {MIN_CV_LENGTH_CHARS}-{MAX_CV_LENGTH_CHARS} chars.", icon="📏")
        
        if text_length >= MIN_CV_LENGTH_CHARS:
            st.success(f"CV Analyzed ({text_length} chars). Searching...")

        # Job Matching Search
        display_matches = []
        all_matches = None
        all_matches_count = 0
        search_error = False
        method_used = "Unknown"

        with st.spinner('🧠 Comparing CV to jobs... Please wait.'):
            try:
                search_top_n = int(os.getenv('TOP_N_RESULTS', '20'))
                all_matches, method_used = find_similar_jobs(cv_text, top_n=search_top_n, active_only=True)

                if all_matches is not None:
                    all_matches_count = len(all_matches)
                    filtered_matches = [j for j in all_matches if isinstance(j.get('score'), (int, float)) 
                                        and j.get('score', 0) >= SIMILARITY_THRESHOLD]
                    filtered_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
                    display_matches = filtered_matches[:MAX_JOBS_TO_DISPLAY]
                    print(f"Search returned {all_matches_count} jobs. Filtered to {len(filtered_matches)} above threshold. Displaying {len(display_matches)}.")
                else:
                    display_matches = []
                    all_matches_count = 0
                    st.warning(f"Search failed or returned no results: {method_used}", icon="🤷")
                    if "Error" in method_used: 
                        search_error = True

            except Exception as find_err:
                st.error("Error during matching.", icon="🔥")
                print(traceback.format_exc())
                display_matches = []
                all_matches = None
                all_matches_count = 0
                search_error = True

        # Create Tabs
        tab_results, tab_feedback = st.tabs(["🎯 Matching Jobs", "📊 Overall Feedback"])

        # Results Tab
        with tab_results:
            if display_matches:
                st.subheader(f"Top {len(display_matches)} Job Matches")
                
                # Calculate filtered results for internal tracking
                filtered_count = len([j for j in (all_matches or []) if isinstance(j.get('score'), (int, float)) 
                                     and j.get('score', 0) >= SIMILARITY_THRESHOLD])
                num_hidden_threshold = max(0, filtered_count - len(display_matches))
                num_below_threshold = max(0, all_matches_count - filtered_count)
                
                # Removed the two caption messages about thresholds and hidden matches

                for i, job in enumerate(display_matches):
                    # Get job ID
                    job_unique_id = job.get('chroma_id')
                    if not job_unique_id:
                        job_unique_id = f"job_fallback_{job.get('Title', 'NoTitle')}_{i}"
                        print(f"WARN: Job missing chroma_id, using fallback: {job_unique_id}")

                    # Get feedback stats
                    job_feedback = feedback_aggregates["per_job"].get(job_unique_id, {"up": 0, "down": 0})

                    with st.container(border=True):
                        col_info, col_score_feedback = st.columns([3, 1.5])
                        
                        with col_info:
                            job_title = job.get('Title', 'N/A')
                            job_company = job.get('Company', 'N/A')
                            job_area = job.get('Area', 'N/A')
                            job_status = job.get('Status', 'unknown').capitalize()
                            job_url = job.get('url', '#')
                            
                            st.markdown(f"**{i+1}. {job_title}**")
                            st.markdown(f"**Company:** {job_company}")
                            st.markdown(f"**Location:** {job_area}")
                            st.markdown(f"**Status:** `{job_status}`")
                            st.markdown("---")
                            
                            if job_url and job_url != '#':
                                st.link_button("Apply Now 🚀", url=job_url, type="primary")
                            else:
                                st.button("Apply Now", disabled=True, help="Application link not available")
                                
                        with col_score_feedback:
                            job_score = job.get('score', 0.0)
                            st.metric("Match Score", f"{job_score:.1f}%", help="CV vs. Job similarity.")
                            st.write("Rate this match:")
                            
                            key_base = f"fb_{job_unique_id}_{i}"
                            is_rated = key_base in st.session_state.feedback_given_jobs
                            fb_buttons_cols = st.columns(2)

                            # Feedback Click Handler
                            def handle_feedback_click(feedback_key, job_id, rating):
                                current_session_id = st.session_state.get('session_id')
                                current_cv_upload_time = st.session_state.get('cv_upload_time')
                                
                                if not current_session_id or not current_cv_upload_time:
                                    st.error("Cannot record feedback: Session info missing.", icon="❌")
                                    return

                                success = record_feedback_hub(
                                    current_session_id,
                                    current_cv_upload_time,
                                    job_id,
                                    rating
                                )
                                
                                if success:
                                    st.session_state.feedback_given_jobs.add(feedback_key)

                            # Feedback buttons
                            with fb_buttons_cols[0]:
                                st.button("👍", key=f"up_{key_base}", help="Good match", disabled=is_rated, 
                                          use_container_width=True,
                                          on_click=handle_feedback_click, args=(key_base, job_unique_id, "up"))
                            with fb_buttons_cols[1]:
                                st.button("👎", key=f"down_{key_base}", help="Not a good match", disabled=is_rated, 
                                          use_container_width=True,
                                          on_click=handle_feedback_click, args=(key_base, job_unique_id, "down"))

                            st.caption(f"Votes: 👍{job_feedback.get('up', 0)} | 👎{job_feedback.get('down', 0)}")
                            if is_rated:
                                st.caption("✔️ Rated")

            # Handle no matches cases
            elif not search_error and all_matches is not None and all_matches_count > 0 and not display_matches:
                st.info(f"Found {all_matches_count} potential matches, but none scored {SIMILARITY_THRESHOLD:.0f}% or higher.", icon="📉")
            elif not search_error and all_matches is not None and all_matches_count == 0:
                st.info("No relevant active jobs found matching your CV's content.", icon="🤷")
            elif all_matches is None and not search_error:
                st.warning("Could not retrieve job matches.", icon="⚠️")

        # Feedback Tab
        with tab_feedback:
            st.subheader("Overall Match Performance & Feedback")
            st.markdown("""
            This section provides insights into how well the job matcher is performing based on **all historical feedback** submitted by users like you.
            Your 👍 and 👎 ratings directly contribute to these aggregated statistics and help us monitor and improve the matching quality over time.
            """)
            
            st.caption(f"(Data sourced anonymously from Hugging Face Dataset: `{HF_DATASET_REPO_ID}`)")
            st.markdown("---")

            total_up = feedback_aggregates.get("total_up", 0)
            total_down = feedback_aggregates.get("total_down", 0)
            total_votes = total_up + total_down

            if total_votes > 0:
                # Key Metrics
                st.markdown("**Key Metrics**")
                satisfaction_score = (total_up / total_votes) * 100 if total_votes > 0 else 0
                col_t, col_s = st.columns(2)
                
                with col_t:
                    st.metric("Total Feedback Votes", total_votes, 
                              help="Total number of 👍 and 👎 ratings received across all jobs.")
                with col_s:
                    st.metric("Overall Satisfaction", f"{satisfaction_score:.1f}%", 
                              help="The percentage of all ratings that were 'Good Match' (👍). Higher is better!")
                
                st.markdown("---")

                # Pie Chart
                st.markdown("**Feedback Distribution**")
                st.markdown("_See the Overall Satisfaction score visualized! This pie chart shows the proportion of 'Good' vs. 'Bad' match ratings._")
                
                pie_data = {'Rating Type': ['Good Matches 👍', 'Bad Matches 👎'], 'Votes': [total_up, total_down]}
                df_pie = pd.DataFrame(pie_data)
                
                try:
                    fig_pie = px.pie(
                        df_pie, 
                        values='Votes', 
                        names='Rating Type', 
                        color='Rating Type',
                        color_discrete_map={'Good Matches 👍':'#2ECC71', 'Bad Matches 👎':'#E74C3C'},
                        labels={'Rating Type':'Feedback'}
                    )
                    fig_pie.update_layout(
                        legend_title_text='Feedback Type', 
                        margin=dict(t=20, b=20, l=0, r=0)
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+value')
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                except Exception as plot_err:
                    st.error(f"Failed to render feedback distribution chart: {plot_err}", icon="📊")
                
                st.markdown("---")

                # Time Series
                st.markdown("**Feedback Trend Over Time**")
                st.markdown("_Track how the number of 'Good' and 'Bad' ratings has changed daily. This helps spot improvements or changes in performance._")
                
                if (feedback_df is not None and not feedback_df.empty and 
                    'timestamp' in feedback_df.columns and 
                    pd.api.types.is_datetime64_any_dtype(feedback_df['timestamp'])):
                    
                    try:
                        feedback_df_sorted = feedback_df.sort_values('timestamp')
                        daily_feedback = feedback_df_sorted.set_index('timestamp').resample('D')['rating'].value_counts().unstack(fill_value=0)
                        
                        if 'up' not in daily_feedback.columns:
                            daily_feedback['up'] = 0
                        if 'down' not in daily_feedback.columns:
                            daily_feedback['down'] = 0
                            
                        daily_feedback = daily_feedback.rename(columns={'up': 'Good 👍', 'down': 'Bad 👎'})

                        if not daily_feedback.empty and (daily_feedback['Good 👍'].sum() > 0 or daily_feedback['Bad 👎'].sum() > 0):
                            fig_time = px.line(
                                daily_feedback, 
                                y=['Good 👍', 'Bad 👎'], 
                                markers=True,
                                labels={"timestamp": "Date", "value": "Number of Ratings", "variable": "Rating Type"},
                                color_discrete_map={'Good 👍': '#2ECC71', 'Bad 👎': '#E74C3C'}
                            )
                            fig_time.update_layout(
                                hovermode="x unified", 
                                legend_title_text='Rating Type', 
                                yaxis_title="Number of Ratings"
                            )
                            st.plotly_chart(fig_time, use_container_width=True)
                        else:
                            st.info("Not enough feedback data points over time to display a trend.", icon="📈")
                            
                    except Exception as time_plot_err:
                        st.error(f"Failed to render feedback trend chart: {time_plot_err}", icon="📈")
                        print(f"Error rendering time series chart: {traceback.format_exc()}")
                        
                else:
                    st.info("Not enough valid timestamp data available to display the feedback trend.", icon="⏳")

            # Handle cases where no feedback exists yet
            elif os.path.exists(LOCAL_FEEDBACK_FILENAME):
                st.info("📊 No feedback votes recorded yet. Rate matches to build these statistics!", icon="✏️")
            else:
                st.info("Feedback system ready. Statistics will appear once users provide ratings.", icon="⏳")

    # Error message already shown by read_cv_file if needed

# Footer
st.divider()
st.caption("CV Job Matcher | MLOps Project Demo")
