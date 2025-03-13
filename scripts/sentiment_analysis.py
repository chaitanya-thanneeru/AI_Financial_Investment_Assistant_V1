from transformers import pipeline

# Load FinBERT Sentiment Model
sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert")

# Sample Financial Headline
text = "Apple faces severe supply chain issues due to China lockdown."

# Run Sentiment Analysis
sentiment = sentiment_pipeline(text)
print("Sentiment:", sentiment)
