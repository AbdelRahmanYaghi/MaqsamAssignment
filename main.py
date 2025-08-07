from src.models import TranscriptionSummary, SENTIMENTS
from src.llm import query_sentiment_llm

from fastapi import FastAPI 

app = FastAPI()

@app.post("/query_sentiment")
def query_sentiment(body: TranscriptionSummary) -> SENTIMENTS:
    return query_sentiment_llm(body.summary).sentiment
