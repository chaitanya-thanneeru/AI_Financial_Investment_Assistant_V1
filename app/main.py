from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

@app.post("/query/")
def get_answer(question: str, context: str):
    answer = qa_pipeline(question=question, context=context)
    return {"answer": answer["answer"]}
