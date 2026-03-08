# 🎙️ Pro-Meet AI: Meeting Intelligence & Action Tracker

An automated intelligence tool that leverages **Natural Language Processing (NLP)** to transform meeting transcripts into structured, actionable insights. This application provides a seamless way to track project progress, assign tasks, and visualize meeting dynamics through a modern web interface.

## 🚀 Key Features

* **Multi-Format Support**: Seamlessly parse `.txt`, `.pdf`, and `.docx` transcripts using pure Python libraries.
* **Automated Summarization**: Generates concise executive summaries using **TF-IDF (Term Frequency-Inverse Document Frequency)** algorithms.
* **Intelligent Action Item Detection**: Automatically identifies tasks, assignees, and deadlines within the conversation.
* **Interactive Dashboard**: A modern **Streamlit** web interface featuring visual analytics for participant activity and keyword frequency.
* **Instant Reporting**: Export processed meeting notes and action plans as professional text reports.

## 🛠️ Tech Stack

* **Python 3.x**
* **Streamlit**: Web interface and dashboard deployment.
* **NLP & Analytics**: Scikit-learn (TF-IDF), Pandas, NumPy.
* **Visualization**: Matplotlib, Seaborn.
* **File Parsing**: pdfplumber (PDF), python-docx (Word).

## 📂 Project Structure

```text
├── app.py                        # Streamlit Web Application (Frontend)
├── meeting_notes_generator.py    # Core NLP Logic (Backend)
├── requirements.txt              # Project Dependencies
└── README.md                     # Project Documentation
```

# 🎙️ Pro-Meet AI: Meeting Intelligence & Action Tracker

**🔗 Live Demo:** [View App Online](https://ai-meeting-notes-generator-nxaihvv42ofjtcxq6cz68k.streamlit.app/)

---

## ⚡ Quick Start

1. Clone the Repository
git clone:
```bash
https://github.com/T786-eng/AI-Meeting-Notes-Generator.git
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Run the Application
```bash
streamlit run app.py
```

📄 License
Distributed under the MIT License.
