import streamlit as st
import pandas as pd
import pdfplumber
from docx import Document
from meeting_notes_generator import MeetingNotesGenerator

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro-Meet AI", page_icon="📝", layout="wide")

# --- PROFESSIONAL UI STYLING ---
# We use custom CSS to create clean, white cards with soft shadows
st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Clean Metric Card Design */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    /* Hover effect for recruiters to notice interactivity */
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        border-color: #4CAF50;
    }

    /* Metric Label (Participants, Action Items, etc.) */
    [data-testid="stMetricLabel"] {
        color: #5f6368 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px;
    }
    
    /* Metric Value (The numbers) */
    [data-testid="stMetricValue"] {
        color: #1a73e8 !important;
        font-weight: 800 !important;
    }

    /* Header styling */
    h1 {
        color: #202124;
        font-weight: 700;
    }
    
    /* Sidebar custom styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

def extract_text(file):
    """Pure Python text extraction without external software dependencies."""
    fname = file.name.lower()
    if fname.endswith('.txt'):
        return file.getvalue().decode("utf-8")
    
    elif fname.endswith('.docx'):
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    
    elif fname.endswith('.pdf'):
        text = ""
        # Use pdfplumber for better text layout extraction
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    return ""

# --- MAIN UI ---
st.title("🎙️ AI Meeting Intelligence")
st.markdown("##### Professional Meeting Analytics & Action Item Extraction")

with st.sidebar:
    st.header("📤 Upload")
    uploaded_file = st.file_uploader("Drop your transcript (TXT, PDF, or DOCX)", type=['txt', 'pdf', 'docx'])
    st.divider()
    st.subheader("⚙️ Analysis Settings")
    summary_len = st.slider("Summary Length (Sentences)", 3, 15, 5)
    
    # Adding a small bio/credit for recruiters
    st.sidebar.markdown("---")
    st.sidebar.caption("🚀 Developed for the **UIDAI Hackathon 2026**")
    st.sidebar.info("This tool uses TF-IDF and NLP for automated summarization and task tracking.")

if uploaded_file:
    try:
        with st.spinner("Processing your meeting transcript..."):
            raw_text = extract_text(uploaded_file)
            
            if not raw_text.strip():
                st.warning("⚠️ No text could be extracted. Please ensure your PDF is text-based and not a scanned image.")
                st.stop()

            # Initialize your core NLP class
            generator = MeetingNotesGenerator()
            generator.load_transcript(text=raw_text)
            generator.extract_participants()
            actions = generator.extract_action_items()

            # --- KEY METRICS DASHBOARD ---
            st.write("### 📊 Meeting Overview")
            m1, m2, m3 = st.columns(3)
            m1.metric("Participants Detected", len(generator.participants))
            m2.metric("Critical Action Items", len(actions))
            m3.metric("Transcript Words", f"{len(raw_text.split()):,}")

            # --- DETAILED ANALYSIS TABS ---
            tab1, tab2, tab3 = st.tabs(["📉 Visual Analytics", "📝 Executive Summary", "✅ Task List"])

            with tab1:
                st.write("#### Sentiment & Keyword Analysis")
                st.pyplot(generator.create_visualizations())

            with tab2:
                st.write("#### Generated Summary")
                st.info(generator.generate_summary(num_sentences=summary_len))
                
                with st.expander("Show Full Raw Text"):
                    st.text_area("", raw_text, height=300)

            with tab3:
                st.write("#### Action Item Tracker")
                if actions:
                    df_actions = pd.DataFrame(actions)
                    # Use st.dataframe for an interactive, sortable table
                    st.dataframe(df_actions, use_container_width=True, hide_index=True)
                else:
                    st.success("🎉 No pending action items were detected in this session!")

            # --- EXPORT SECTION ---
            st.divider()
            st.download_button(
                label="📥 Download Full Meeting Report",
                data=f"MEETING SUMMARY:\n{generator.generate_summary()}\n\nACTION ITEMS:\n{str(actions)}",
                file_name=f"Report_{uploaded_file.name}.txt",
                mime="text/plain"
            )

    except Exception as e:
        st.error(f"Analysis Error: {e}")
        st.info("Check if your transcript format is standard or contains unusual characters.")
else:
    # Display a professional welcome message if no file is uploaded
    st.info("👋 **Welcome!** Please upload a transcript in the sidebar to begin generating insights.")