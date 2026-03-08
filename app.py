import streamlit as st
from meeting_notes_generator import MeetingNotesGenerator
import io

# Page Config
st.set_page_config(page_title="AI Meeting Assistant", page_icon="🎙️")

st.title("🎙️ AI Meeting Notes & Action Items")
st.markdown("Upload your meeting transcript to automatically generate summaries and visualize tasks.")

# Sidebar for inputs
st.sidebar.header("Settings")
num_summary = st.sidebar.slider("Summary Length (Sentences)", 3, 10, 5)

# File Upload
uploaded_file = st.file_uploader("Choose a transcript file (.txt)", type="txt")

if uploaded_file:
    # 1. Process Data
    text = uploaded_file.getvalue().decode("utf-8")
    generator = MeetingNotesGenerator()
    generator.load_transcript(text=text)
    generator.extract_participants()
    generator.extract_action_items()

    # 2. Display Analytics
    st.header("📊 Meeting Insights")
    fig = generator.create_visualizations()
    st.pyplot(fig)

    # 3. Display Summary
    st.header("📝 Executive Summary")
    summary = generator.generate_summary(num_sentences=num_summary)
    st.write(summary)

    # 4. Action Items Table
    st.header("✅ Action Items")
    if generator.action_items:
        df_actions = pd.DataFrame(generator.action_items)
        st.table(df_actions)
    else:
        st.info("No clear action items detected.")

    # 5. Download Report
    st.divider()
    report_data = f"MEETING REPORT\n---\nSUMMARY:\n{summary}\n\nACTION ITEMS:\n"
    for item in generator.action_items:
        report_data += f"- {item['action']} (Assignee: {item['assignee']})\n"
    
    st.download_button(
        label="Download Full Report",
        data=report_data,
        file_name="meeting_report.txt",
        mime="text/plain"
    )

else:
    st.info("Please upload a .txt file to begin analysis.")