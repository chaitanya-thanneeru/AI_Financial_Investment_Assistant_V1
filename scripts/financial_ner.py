from transformers import pipeline

# Load FinBERT NER Model
ner_pipeline = pipeline("ner", model="xlm-roberta-large-finetuned-conll03-english")

# Sample Financial Text
text = "Tesla reported Q3 revenue of $21.45 billion, a 30% increase from last year."

# Run NER
entities = ner_pipeline(text)
print("Named Entities:", entities)
