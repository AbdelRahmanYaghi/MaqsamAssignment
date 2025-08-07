from src.models import SentimentResult

from openai import OpenAI
from tqdm import tqdm

import json
import csv
import os

MODEL_NAME = input("Please enter the name of the model (From Ollama): ")
# MODEL_NAME = "hf.co/tensorblock/Phi-4-mini-instruct-abliterated-GGUF:Q8_0"
# MODEL_NAME = "deepseek-r1:8b"

URL = "http://localhost:11434/v1" # GPT-GENERATED >> Forgot that I have to put /v1 to make it work
HEADER = "Content-Type: application/json"
SYS_PROMPT = open('src/prompts/SENTIMENT_ANALYSIS_SYS_PROMPT.md', 'r').read()
RESULTS_PATH = os.path.join("llm_tests", "test_results")

os.makedirs(RESULTS_PATH, exist_ok=True)

client = OpenAI(base_url=URL, api_key='ollama')

with open(os.path.join("llm_tests", "call_summaries_sentiment.csv"), "r") as f:
    # [1:] => Skip column names
    reader = csv.reader(f.readlines()[1:])
    
results = {'Arabic': [], 'English': []}
for row in tqdm(reader, total=30):
    out = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": row[0]}
        ],
        response_format=SentimentResult
    )
    
    results[row[2]].append(
        {
            'summary': row[0],
            'predicted_sentiment': json.loads(out.choices[0].message.content)['sentiment'],
            'true_sentiment': row[1],
            'justification':json.loads(out.choices[0].message.content)['sentiment_justification'],
        }
    )

with open(os.path.join(RESULTS_PATH, f"{MODEL_NAME.replace("/", "")}.json"), 'w') as f:
    json.dump(results, f, indent=4)
