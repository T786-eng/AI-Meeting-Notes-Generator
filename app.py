import streamlit as st
import pandas as pd
import pdfplumber
from docx import Document
from meeting_notes_generator import MeetingNotesGenerator

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro-Meet AI", page_icon="📝", layout="wide")

# --- PROFESSIONAL UI STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        border-color: #4CAF50;
    }
    [data-testid="stMetricLabel"] { color: #5f6368 !important; font-weight: 600 !important; }
    [data-testid="stMetricValue"] { color: #1a73e8 !important; font-weight: 800 !important; }
    h1 { color: #202124; font-weight: 700; }
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

def extract_text(file):
    fname = file.name.lower()
    if fname.endswith('.txt'):
        return file.getvalue().decode("utf-8")
    elif fname.endswith('.docx'):
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif fname.endswith('.pdf'):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    return ""

# --- MAIN UI ---
st.title("🎙️ AI Meeting Intelligence")
st.markdown("##### Automated Meeting Analytics & Task Tracking System")

with st.sidebar:
    st.header("📤 Upload Center")
    uploaded_file = st.file_uploader("Upload Transcript (TXT, PDF, DOCX)", type=['txt', 'pdf', 'docx'])
    st.divider()
    st.subheader("⚙️ Analysis Settings")
    summary_len = st.slider("Summary Length (Sentences)", 3, 15, 5)

if uploaded_file:
    try:
        with st.spinner("Analyzing transcript..."):
            raw_text = extract_text(uploaded_file)
            
            if not raw_text.strip():
                st.warning("⚠️ Could not extract text. Ensure the file is text-based.")
                st.stop()

            generator = MeetingNotesGenerator()
            generator.load_transcript(text=raw_text)
            generator.extract_participants()
            actions = generator.extract_action_items()

            # --- KEY METRICS ---
            st.write("### 📊 Meeting Overview")
            m1, m2, m3 = st.columns(3)
            m1.metric("Participants", len(generator.participants))
            m2.metric("Action Items", len(actions))
            m3.metric("Word Count", f"{len(raw_text.split()):,}")

            # --- TABS ---
            tab1, tab2, tab3 = st.tabs(["📉 Visual Analytics", "📝 Summary", "✅ Tasks"])

            with tab1:
                st.pyplot(generator.create_visualizations())

            with tab2:
                st.info(generator.generate_summary(num_sentences=summary_len))
                with st.expander("Show Full Transcript"):
                    st.text_area("", raw_text, height=300)

            with tab3:
                if actions:
                    st.dataframe(pd.DataFrame(actions), use_container_width=True, hide_index=True)
                else:
                    st.success("No pending action items detected.")

            st.divider()
            st.download_button(
                label="📥 Download Report",
                data=f"SUMMARY:\n{generator.generate_summary()}\n\nTASKS:\n{str(actions)}",
                file_name=f"Meeting_Report.txt",
                mime="text/plain"
            )
    except Exception as e:
        st.error(f"Analysis Error: {e}")
else:
    st.info("👋 Upload a transcript in the sidebar to begin.")