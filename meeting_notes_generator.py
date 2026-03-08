"""
AI Meeting Notes & Action Items Generator
==========================================
Upload meeting transcripts and automatically generate:
- Meeting summary
- Action items with assignees and priorities
- Visualizations of meeting insights
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Use a standard style compatible with most servers
plt.style.use('ggplot') 

class MeetingNotesGenerator:
    def __init__(self):
        self.transcript = ""
        self.sentences = []
        self.action_items = []
        self.participants = []
        
    def load_transcript(self, text=None):
        self.transcript = text
        self.sentences = [s.strip() for s in re.split(r'[.!?]+', self.transcript) if s.strip()]
        return self
    
    def extract_participants(self):
        name_patterns = re.findall(r'\b([A-Z][a-z]+)(?:\s+[A-Z][a-z]+)?:', self.transcript)
        self.participants = list(set(name_patterns))
        return self.participants
    
    def generate_summary(self, num_sentences=5):
        if not self.sentences: return ""
        if len(self.sentences) < num_sentences: num_sentences = len(self.sentences)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix = vectorizer.fit_transform(self.sentences)
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        top_indices = sorted(sentence_scores.argsort()[-num_sentences:][::-1])
        return '. '.join([self.sentences[i] for i in top_indices]) + '.'
    
    def extract_action_items(self):
        action_keywords = ['will', 'should', 'need to', 'must', 'have to', 'task', 'follow up', 'deadline']
        self.action_items = []
        for sentence in self.sentences:
            if any(kw in sentence.lower() for kw in action_keywords):
                assignee = "Unassigned"
                for p in self.participants:
                    if p.lower() in sentence.lower(): assignee = p; break
                
                priority = "High" if any(w in sentence.lower() for w in ['urgent', 'asap']) else "Medium"
                deadline_match = re.search(r'by\s+(\w+\s+\d+|next\s+\w+|end\s+of\s+\w+)', sentence.lower())
                
                self.action_items.append({
                    'action': sentence,
                    'assignee': assignee,
                    'priority': priority,
                    'deadline': deadline_match.group(1) if deadline_match else "No deadline"
                })
        return self.action_items

    def analyze_keywords(self, top_n=10):
        words = re.findall(r'\b[a-z]{4,}\b', self.transcript.lower())
        stop_words = set(['have', 'will', 'should', 'could', 'would', 'this', 'that', 'with'])
        words = [w for w in words if w not in stop_words]
        return Counter(words).most_common(top_n)

    def create_visualizations(self):
        fig = plt.figure(figsize=(12, 8))
        
        # Action Items Priority
        ax1 = plt.subplot(2, 2, 1)
        if self.action_items:
            df = pd.DataFrame(self.action_items)
            df['priority'].value_counts().plot(kind='bar', ax=ax1, color=['#ff6b6b', '#ffd93d'])
        ax1.set_title('Action Items by Priority')

        # Keywords
        ax2 = plt.subplot(2, 2, 2)
        top_k = self.analyze_keywords(10)
        if top_k:
            keys, vals = zip(*top_k)
            ax2.barh(keys, vals, color='#4ecdc4')
            ax2.invert_yaxis()
        ax2.set_title('Top Keywords')

        # Statistics Text Box
        ax3 = plt.subplot(2, 2, 3)
        ax3.axis('off')
        stats = f"Sentences: {len(self.sentences)}\nWords: {len(self.transcript.split())}\nActions: {len(self.action_items)}"
        ax3.text(0.1, 0.5, stats, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig # Crucial for Streamlit!