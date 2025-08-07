from pydantic import BaseModel
from typing import Literal

class SentimentResult(BaseModel):
    """ Sentiment output format for the model.
    
        We are including a sentiment_justification despite not actually using it.
        The reason behind this is that it is proven that having an LLM judge itself could
        - in many cases drasticly improve its performance. This paper goes over the details
        - of it:
        
        ```Trivedi, P., Gulati, A., Molenschot, O., Rajeev, M. A., Ramamurthy, R., Stevens, K., ... & Rajani, N. (2024). Self-rationalization improves llm as a fine-grained judge. arXiv preprint arXiv:2410.05495.```
    """
    sentiment: Literal[ "Positive", "Negative", "Neutral"]
    sentiment_justification: str