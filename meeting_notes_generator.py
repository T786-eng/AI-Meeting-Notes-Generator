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
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MeetingNotesGenerator:
    """Main class for processing meeting transcripts"""
    
    def __init__(self):
        self.transcript = ""
        self.sentences = []
        self.action_items = []
        self.participants = []
        
    def load_transcript(self, file_path=None, text=None):
        """Load meeting transcript from file or text"""
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.transcript = f.read()
        elif text:
            self.transcript = text
        else:
            raise ValueError("Please provide either file_path or text")
        
        # Split into sentences
        self.sentences = [s.strip() for s in re.split(r'[.!?]+', self.transcript) if s.strip()]
        print(f"✓ Transcript loaded: {len(self.sentences)} sentences")
        return self
    
    def extract_participants(self):
        """Extract participant names from transcript"""
        # Look for patterns like "John:", "Sarah said", etc.
        name_patterns = re.findall(r'\b([A-Z][a-z]+)(?:\s+[A-Z][a-z]+)?:', self.transcript)
        self.participants = list(set(name_patterns))
        print(f"✓ Found {len(self.participants)} participants: {', '.join(self.participants)}")
        return self.participants
    
    def generate_summary(self, num_sentences=5):
        """Generate extractive summary using TF-IDF"""
        if len(self.sentences) < num_sentences:
            num_sentences = len(self.sentences)
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix = vectorizer.fit_transform(self.sentences)
        
        # Calculate sentence scores based on TF-IDF
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Get top sentences
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_indices = sorted(top_indices)  # Maintain order
        
        summary_sentences = [self.sentences[i] for i in top_indices]
        summary = '. '.join(summary_sentences) + '.'
        
        print(f"✓ Generated summary with {num_sentences} key sentences")
        return summary
    
    def extract_action_items(self):
        """Extract action items using keyword matching"""
        action_keywords = [
            'will', 'should', 'need to', 'must', 'have to', 'going to',
            'action item', 'todo', 'task', 'follow up', 'assign',
            'responsible for', 'deadline', 'by end of', 'complete'
        ]
        
        action_items = []
        
        for sentence in self.sentences:
            sentence_lower = sentence.lower()
            
            # Check if sentence contains action keywords
            if any(keyword in sentence_lower for keyword in action_keywords):
                # Extract potential assignee
                assignee = "Unassigned"
                for participant in self.participants:
                    if participant.lower() in sentence_lower:
                        assignee = participant
                        break
                
                # Determine priority based on urgency keywords
                priority = "Medium"
                if any(word in sentence_lower for word in ['urgent', 'asap', 'immediately', 'critical']):
                    priority = "High"
                elif any(word in sentence_lower for word in ['when possible', 'eventually', 'low priority']):
                    priority = "Low"
                
                # Extract deadline if present
                deadline_match = re.search(r'by\s+(\w+\s+\d+|\d+/\d+|next\s+\w+|end\s+of\s+\w+)', sentence_lower)
                deadline = deadline_match.group(1) if deadline_match else "No deadline"
                
                action_items.append({
                    'action': sentence,
                    'assignee': assignee,
                    'priority': priority,
                    'deadline': deadline
                })
        
        self.action_items = action_items
        print(f"✓ Extracted {len(action_items)} action items")
        return action_items
    
    def get_action_items_dataframe(self):
        """Convert action items to pandas DataFrame"""
        if not self.action_items:
            self.extract_action_items()
        
        df = pd.DataFrame(self.action_items)
        return df
    
    def analyze_keywords(self, top_n=10):
        """Extract and count top keywords from transcript"""
        # Remove common words and split
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                         'of', 'with', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has',
                         'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may',
                         'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
                         'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where',
                         'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
                         'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                         'so', 'than', 'too', 'very', 's', 't', 'just', 'now'])
        
        words = re.findall(r'\b[a-z]{4,}\b', self.transcript.lower())
        words = [w for w in words if w not in stop_words]
        
        word_freq = Counter(words)
        top_keywords = word_freq.most_common(top_n)
        
        return top_keywords
    
    def create_visualizations(self, output_prefix='meeting_analysis'):
        """Create comprehensive visualizations of meeting analysis"""
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Action Items by Priority
        ax1 = plt.subplot(2, 3, 1)
        if self.action_items:
            df = self.get_action_items_dataframe()
            priority_counts = df['priority'].value_counts()
            colors = {'High': '#ff6b6b', 'Medium': '#ffd93d', 'Low': '#6bcf7f'}
            bars = ax1.bar(priority_counts.index, priority_counts.values, 
                          color=[colors.get(x, '#95a5a6') for x in priority_counts.index])
            ax1.set_title('Action Items by Priority', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Count')
            ax1.set_xlabel('Priority')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        else:
            ax1.text(0.5, 0.5, 'No action items found', ha='center', va='center')
            ax1.set_title('Action Items by Priority', fontsize=12, fontweight='bold')
        
        # 2. Action Items by Assignee
        ax2 = plt.subplot(2, 3, 2)
        if self.action_items:
            assignee_counts = df['assignee'].value_counts().head(8)
            bars = ax2.barh(assignee_counts.index, assignee_counts.values, color='#4ecdc4')
            ax2.set_title('Action Items by Assignee', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Count')
            ax2.set_ylabel('Assignee')
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax2.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{int(width)}', ha='left', va='center', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No action items found', ha='center', va='center')
            ax2.set_title('Action Items by Assignee', fontsize=12, fontweight='bold')
        
        # 3. Top Keywords
        ax3 = plt.subplot(2, 3, 3)
        top_keywords = self.analyze_keywords(10)
        if top_keywords:
            keywords, counts = zip(*top_keywords)
            bars = ax3.barh(keywords, counts, color='#95e1d3')
            ax3.set_title('Top 10 Keywords in Meeting', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Frequency')
            ax3.invert_yaxis()
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax3.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{int(width)}', ha='left', va='center', fontsize=9)
        
        # 4. Sentence Length Distribution
        ax4 = plt.subplot(2, 3, 4)
        sentence_lengths = [len(s.split()) for s in self.sentences]
        ax4.hist(sentence_lengths, bins=20, color='#a8e6cf', edgecolor='black', alpha=0.7)
        ax4.axvline(np.mean(sentence_lengths), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(sentence_lengths):.1f} words')
        ax4.set_title('Sentence Length Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Words per Sentence')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        # 5. Meeting Statistics
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        
        stats_text = f"""
        📊 MEETING STATISTICS
        {'='*35}
        
        Total Sentences: {len(self.sentences)}
        Total Words: {len(self.transcript.split())}
        Avg Sentence Length: {np.mean(sentence_lengths):.1f} words
        
        Participants: {len(self.participants)}
        Action Items: {len(self.action_items)}
        
        Priority Breakdown:
        """
        
        if self.action_items:
            for priority in ['High', 'Medium', 'Low']:
                count = len([a for a in self.action_items if a['priority'] == priority])
                stats_text += f"  • {priority}: {count}\n        "
        
        ax5.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        # 6. Word Cloud Alternative - Top Words Bubble Chart
        ax6 = plt.subplot(2, 3, 6)
        top_keywords = self.analyze_keywords(15)
        if top_keywords:
            keywords, counts = zip(*top_keywords)
            x = np.random.rand(len(keywords))
            y = np.random.rand(len(keywords))
            sizes = np.array(counts) * 50
            
            scatter = ax6.scatter(x, y, s=sizes, alpha=0.6, c=range(len(keywords)), 
                                 cmap='viridis', edgecolors='black', linewidth=1)
            
            for i, keyword in enumerate(keywords):
                ax6.annotate(keyword, (x[i], y[i]), ha='center', va='center',
                           fontsize=8, fontweight='bold')
            
            ax6.set_title('Keyword Importance Visualization', fontsize=12, fontweight='bold')
            ax6.set_xlim(-0.1, 1.1)
            ax6.set_ylim(-0.1, 1.1)
            ax6.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_file = f'{output_prefix}_visualization.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Visualizations saved to {output_file}")
        
        return fig
    
    def generate_report(self, output_file='meeting_report.txt'):
        """Generate a comprehensive text report"""
        report = []
        report.append("="*60)
        report.append("           MEETING NOTES & ACTION ITEMS REPORT")
        report.append("="*60)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Participants
        report.append("\n📋 PARTICIPANTS")
        report.append("-" * 60)
        if self.participants:
            for i, participant in enumerate(self.participants, 1):
                report.append(f"{i}. {participant}")
        else:
            report.append("No participants identified")
        
        # Summary
        report.append("\n\n📝 MEETING SUMMARY")
        report.append("-" * 60)
        summary = self.generate_summary()
        report.append(summary)
        
        # Action Items
        report.append("\n\n✅ ACTION ITEMS")
        report.append("-" * 60)
        if self.action_items:
            for i, item in enumerate(self.action_items, 1):
                report.append(f"\n{i}. {item['action']}")
                report.append(f"   • Assignee: {item['assignee']}")
                report.append(f"   • Priority: {item['priority']}")
                report.append(f"   • Deadline: {item['deadline']}")
        else:
            report.append("No action items identified")
        
        # Keywords
        report.append("\n\n🔑 KEY TOPICS DISCUSSED")
        report.append("-" * 60)
        top_keywords = self.analyze_keywords(10)
        for keyword, count in top_keywords:
            report.append(f"• {keyword.capitalize()}: mentioned {count} times")
        
        report.append("\n" + "="*60)
        report.append("                    END OF REPORT")
        report.append("="*60)
        
        # Save report
        report_text = '\n'.join(report)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"✓ Report saved to {output_file}")
        return report_text


def main():
    """Main function to demonstrate usage"""
    
    # Sample meeting transcript
    sample_transcript = """
    John: Good morning everyone. Let's start today's project meeting. We need to discuss the Q1 deliverables.
    
    Sarah: Thanks John. I've completed the market analysis. The results show a 15% increase in customer engagement.
    We should focus on mobile optimization next quarter.
    
    Mike: That's great Sarah. I will work on the mobile app updates by end of February. 
    We need to prioritize the user interface improvements urgently.
    
    John: Excellent. Mike, please coordinate with the design team. Sarah, can you prepare a detailed report 
    by next Friday? We'll need it for the stakeholder meeting.
    
    Sarah: Absolutely. I will have the comprehensive report ready. Should I include the competitor analysis too?
    
    John: Yes, please include that. It's critical for our strategy. Mike, you must also complete the API integration 
    before the mobile launch.
    
    Mike: Understood. I'll follow up with the backend team this week. We need to ensure all security protocols 
    are in place.
    
    Lisa: I can help with testing. I will create test cases and have them ready by Wednesday.
    
    John: Perfect. Let's schedule a follow-up meeting for next Monday to review progress. Action items are clear?
    
    Everyone: Yes!
    
    John: Great. Meeting adjourned. Thank you all for your contributions to this important project.
    """
    
    print("\n🚀 AI Meeting Notes & Action Items Generator")
    print("=" * 60)
    
    # Initialize generator
    generator = MeetingNotesGenerator()
    
    # Load transcript
    generator.load_transcript(text=sample_transcript)
    
    # Extract participants
    generator.extract_participants()
    
    # Extract action items
    generator.extract_action_items()
    
    # Generate visualizations
    print("\n📊 Creating visualizations...")
    generator.create_visualizations()
    
    # Generate report
    print("\n📄 Generating report...")
    report = generator.generate_report()
    
    print("\n" + "="*60)
    print("✅ PROCESSING COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  • meeting_analysis_visualization.png")
    print("  • meeting_report.txt")
    print("\n💡 To use your own transcript:")
    print("  generator.load_transcript(file_path='your_transcript.txt')")
    print("  or")
    print("  generator.load_transcript(text='your transcript text here')")
    

if __name__ == "__main__":
    main()