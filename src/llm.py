from src.models import SentimentResult, SentimentResultWithJustification
from src.log_config import get_logger

from openai import OpenAI

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

client = OpenAI(base_url=os.getenv("LLM_URL"), api_key='ollama')
logger.info("Successfully initialized Ollama model")

def query_sentiment_llm(summary: str, model_name: str = "qwen3:8b", return_justification: bool = False) -> SentimentResult:
    '''
    Passes a transcription summary to an LLM with a system prompt to classify its sentiment.
    
    Args:
        summary (str): The string containing the summary.
        return_justification (bool): If the justification it needed from the model, set it to true.
        
    Return:
        sentiment ("Positive" | "Negative" | "Neutral"): The sentiment classified by the model.
        sentiment_justification: (Optional[str]): Justification for the sentiment classified.
        
    Raises:
        ...
    '''
    URL = "http://localhost:11434/v1" # GPT-GENERATED >> Forgot that I have to put /v1 to make it work

    client = OpenAI(base_url=URL, api_key='ollama')
        
    try:
        out = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": _SYS_PROMPT},
                {"role": "user", "content": summary}
            ],
            response_format=SentimentResultWithJustification
        )
    except ConnectionError as e:
        logger.error(f"Failed reaching the Ollama server. {e}")    
        raise ConnectionError("Faced an error sending a request to LLM server.")
    except TimeoutError as e:
        logger.error(f"Timed out while reaching the Ollama server. {e}")    
        raise TimeoutError("Request timed out.")
    except Exception as e:
        logger.error(f"Unexpected LLM request error. {e}")
        raise RuntimeError(f"Unexpected LLM request. Something went wrong. {e}")

    if return_justification:
        return SentimentResultWithJustification(**json.loads(out.choices[0].message.content))
    
    return SentimentResult(sentiment=json.loads(out.choices[0].message.content)['sentiment'])