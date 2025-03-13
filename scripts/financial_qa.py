from transformers import pipeline

# Load QA Model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Sample Financial Document
context = """
Tesla's Q3 revenue was $21.45 billion, marking a 30% increase YoY.
Net income stood at $3.29 billion, with strong EV demand.
"""

# Ask AI a Question
question = "What was Tesla's Q3 revenue?"
answer = qa_pipeline(question=question, context=context)

print("Answer:", answer["answer"])
