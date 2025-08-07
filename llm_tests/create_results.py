from src.llm import query_sentiment_llm

from tqdm import tqdm

import json
import csv
import os

MODEL_NAME = input("Please enter the name of the model (From Ollama): ")

RESULTS_PATH = os.path.join("llm_tests", "test_results")

os.makedirs(RESULTS_PATH, exist_ok=True)

with open(os.path.join("llm_tests", "call_summaries_sentiment.csv"), "r") as f:
    # [1:] => Skip column names
    reader = csv.reader(f.readlines()[1:])
    
results = {'Arabic': [], 'English': []}

for row in tqdm(reader, total=30):
    out = query_sentiment_llm(row[0], model_name=MODEL_NAME, return_justification=True)
    
    results[row[2]].append(
        {
            'summary': row[0],
            'predicted_sentiment': out.sentiment,
            'true_sentiment': row[1],
            'justification': out.sentiment_justification,
        }
    )

with open(os.path.join(RESULTS_PATH, f"{MODEL_NAME.replace("/", "")}.json"), 'w') as f:
    json.dump(results, f, indent=4)
