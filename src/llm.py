from src.models import SentimentResult, SentimentResultWithJustification

from openai import OpenAI

import json

# Loaded globally so that it doesn't load every time the function is called
_SYS_PROMPT = open('src/prompts/SENTIMENT_ANALYSIS_SYS_PROMPT.md', 'r').read()

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
    except Exception as conn_e:
        raise ConnectionError(f"Faced an error sending a request to OpenAI. Error: {conn_e}")

    if return_justification:
        return SentimentResultWithJustification(**json.loads(out.choices[0].message.content))
    
    return SentimentResult(sentiment=json.loads(out.choices[0].message.content)['sentiment'])