from dotenv import load_dotenv
load_dotenv()

from src.models import TranscriptionSummary, SENTIMENTS
from src.llm import query_sentiment_llm, verify_model_pulled
from src.log_config import setup_logging, get_logger

from contextlib import asynccontextmanager
from fastapi import FastAPI 

import os

setup_logging()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Application startup: Initializing resources...")
    verify_model_pulled(os.getenv("OLLAMA_MODEL"))

    yield

    print("Application shutdown: Cleaning up resources...")

app = FastAPI(lifespan=lifespan)

@app.get("/verify_model")
def download_model(model_name):
    return verify_model_pulled(model_name)

@app.post("/query_sentiment")
def query_sentiment(body: TranscriptionSummary) -> SENTIMENTS:
    '''
    Endpoint function to classify the sentiment of a transcription summary.
    
    Args:
        body (TranscriptionSummary): Body of the request, contains the summary of the transcription.
        
    Returns:
        SENTIMENTS ("Positive" | "Negative" | "Neutral"): The sentiment determined by the model.
    '''
    logger.info(f"Processing {body.summary[:100]}")
    
    try:
        return query_sentiment_llm(body.summary).sentiment
    except TimeoutError as e:
        logger.error(f"Timeout Error: {e}", exc_info=True)
        raise Exception("Either the given summary was too large, or the model is currently handling many other requests.")
    except ConnectionError as e:
        logger.error(f"Connnection Error: {e}", exc_info=True)
        raise Exception("Failed to each the LLM server.")
    except Exception as e:
        logger.error(f"Faced an unhandled Error: {e}", exc_info=True)
        raise RuntimeError(f"Faced an unhandled Error: {e}")
        
