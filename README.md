# AI Meeting Notes & Action Items Generator

An automated tool that uses Natural Language Processing (NLP) to parse meeting transcripts. It generates concise summaries, extracts actionable tasks with priorities, and creates visual analytics of the meeting dynamics.

## 🚀 Features

* **Automated Summarization**: Extracts key insights using TF-IDF algorithms.
* **Action Item Detection**: Identifies tasks, assignees, deadlines, and priority levels (High/Medium/Low).
* **Visual Analytics**: Generates charts for keyword frequency, action item distribution, and participant activity.
* **Report Generation**: Exports a formatted text file (`meeting_report.txt`) ready for distribution.

## 🛠️ Tech Stack

* **Python 3.x**
* **Pandas & NumPy**: Data manipulation and processing.
* **Scikit-learn**: Text feature extraction (TF-IDF).
* **Matplotlib & Seaborn**: Data visualization.

## 📂 Project Structure

```text
├── meeting_notes_generator.py    # Main script
├── requirements.txt              # Dependencies
├── meeting_report.txt            # Output: Text report
└── meeting_analysis.png          # Output: Visual dashboard