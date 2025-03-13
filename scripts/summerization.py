from transformers import pipeline

# Load Summarization Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Sample Earnings Report
report = """
Tesla reported total revenue of $21.45 billion, up 30% YoY. Net income stood at $3.29 billion, 
driven by strong EV sales. However, production costs increased due to supply chain constraints.
"""

# Summarize
summary = summarizer(report, max_length=100, min_length=30, do_sample=False)
print("Summary:", summary)
