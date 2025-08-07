from pydantic import BaseModel

from typing import Literal, Optional

SENTIMENTS = Literal[ "Positive", "Negative", "Neutral"]

class TranscriptionSummary(BaseModel):
    """ Transcription summary model for fastapi post. """
    summary: str
    
class SentimentResult(BaseModel):
    """ Sentiments """
    sentiment: SENTIMENTS
    
class SentimentResultWithJustification(SentimentResult):
    """ Sentiment output format for the model.
    
        We are including a sentiment_justification despite not actually using it.
        The reason behind this is that it is proven that having an LLM judge itself could
        - in many cases drasticly improve its performance. This paper goes over the details
        - of it:
        
        ```Trivedi, P., Gulati, A., Molenschot, O., Rajeev, M. A., Ramamurthy, R., Stevens, K., ... & Rajani, N. (2024). Self-rationalization improves llm as a fine-grained judge. arXiv preprint arXiv:2410.05495.```
    """
    sentiment_justification: Optional[str]
