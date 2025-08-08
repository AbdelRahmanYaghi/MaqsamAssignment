from src.models import SentimentResult, SentimentResultWithJustification
from src.log_config import get_logger
from src.helper import process_download_streaming_response

from openai import OpenAI
import requests

import json
import os

logger = get_logger(__name__)

# Loaded globally so that it doesn't load every time the function is called
if os.path.exists('src/prompts/SENTIMENT_ANALYSIS_SYS_PROMPT.md'):
    with open('src/prompts/SENTIMENT_ANALYSIS_SYS_PROMPT.md', 'r') as f:
        _SYS_PROMPT = f.read()
else:
    logger.error(f"Couldn't find system prompt at src/prompts/SENTIMENT_ANALYSIS_SYS_PROMPT.md", exc_info=True)
    raise FileExistsError("Couldn't find SENTIMENT_ANALYSIS_SYS_PROMPT.md in src/prompts")

client = OpenAI(base_url=os.getenv("LLM_URL") + "/v1", api_key='ollama')
logger.info("Successfully initialized Ollama model")

def verify_model_pulled(model_name: str) -> dict:
    '''
    Given an Ollama model's name, the function verifies its downloaded.
    If the model is not donwloaded, then it is downloaded.
    
    Args:
        model_name (str): The name of the Ollama model.
        
    Return:
        message (str): Status whether the model has been donwload yet.
        
    '''
        
    downloaded_models = requests.get(os.getenv("LLM_URL") + "/api/tags").json()
    if model_name not in [model['name'] for model in downloaded_models['models']]:
        logger.info(f"Model {model_name} not found. Downloading...")
        
        
        try:
            response = requests.post(
                url=os.getenv("LLM_URL") + "/api/pull",
                data=json.dumps({"model":model_name}),
                headers={"Content-Type": "application/json"},
                stream=True)
            
            return process_download_streaming_response(response, model_name)
                
        except (requests.RequestException, TimeoutError, ConnectionError) as e:
            logger.error(f"Failed to query LLM model {model_name}: {e}", exc_info=True)
            raise
              
    else:
        return {"message": "Model already downloaded"}

def query_sentiment_llm(summary: str, model_name: str = os.getenv("OLLAMA_MODEL"), return_justification: bool = False) -> SentimentResult:
    '''
    Passes a transcription summary to an LLM with a system prompt to classify its sentiment.
    
    Args:
        summary (str): The string containing the summary.
        return_justification (bool): If the justification it needed from the model, set it to true.
        
    Return:
        sentiment (str["Positive" | "Negative" | "Neutral"]): The sentiment classified by the model.
        sentiment_justification: (Optional[str]): Justification for the sentiment classified.
    '''    
    try:
        out = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": _SYS_PROMPT},
                {"role": "user", "content": summary}
            ],
            response_format=SentimentResultWithJustification
        )
    except (requests.RequestException, TimeoutError, ConnectionError) as e:
        logger.error(f"Failed to query LLM model {model_name}: {e}", exc_info=True)
        raise
    
    if return_justification:
        return SentimentResultWithJustification(**json.loads(out.choices[0].message.content))
    
    return SentimentResult(sentiment=json.loads(out.choices[0].message.content)['sentiment'])