import json
import os

RESULTS_PATH = os.path.join("llm_tests", "test_results")

models_results_name = os.listdir(RESULTS_PATH)

distance_mappings = { 'Positive' : 1, 'Neutral' : 0, 'Negative' : -1 }

# These 4 supporting functions were GPT generated
def label_set(model_results):
    """Returns the set of unique labels in the dataset."""
    return set(row['true_sentiment'] for row in model_results)

def true_positives(label, model_results):
    """Correctly predicted as `label`."""
    return sum(1 for row in model_results if row['true_sentiment'] == label and row['predicted_sentiment'] == label)

def predicted_positives(label, model_results):
    """Predicted as `label`, regardless of ground truth."""
    return sum(1 for row in model_results if row['predicted_sentiment'] == label)

def actual_positives(label, model_results):
    """Actually were `label`, regardless of prediction."""
    return sum(1 for row in model_results if row['true_sentiment'] == label)
# ################################################

criterias = {
    "Accuracy": lambda model_results: 
        sum([row['true_sentiment'] == row['predicted_sentiment'] 
            for row in model_results]) 
        / len(model_results),
    
    # Distance, where Positive = 1, Neutral = 0, Negative = -1.
    "Distance": lambda model_results: sum([
        abs(distance_mappings[row['true_sentiment']] - distance_mappings[row['predicted_sentiment']])
            for row in model_results]),
 
    # The precision and recall were GPT-generated
    "Precision": lambda model_results: (
        sum([
            true_positives(label, model_results) / predicted_positives(label, model_results)
            if predicted_positives(label, model_results) > 0 else 0.0
            for label in label_set(model_results)
        ]) / len(label_set(model_results))
    ),

    "Recall": lambda model_results: (
        sum([
            true_positives(label, model_results) / actual_positives(label, model_results)
            if actual_positives(label, model_results) > 0 else 0.0
            for label in label_set(model_results)
        ]) / len(label_set(model_results))
    ) 
}

model_scores = {}

for model_results_name in models_results_name:
    model_results = json.load(open(os.path.join(RESULTS_PATH, model_results_name), 'r'))
    
    eng_scores = {}
    ara_scores = {}
        
    for criteria_title, criteira_function in criterias.items():
        eng_scores[criteria_title] = criteira_function(model_results['English'])
        ara_scores[criteria_title] = criteira_function(model_results['Arabic'])
        
    model_scores[model_results_name] = {
        "English Scores": eng_scores,
        "Arabic Scores": ara_scores,
        "Mixed Scores": {criteria_title: (ara_criteria_score + eng_criteria_score)/2 for ((criteria_title, ara_criteria_score), (_, eng_criteria_score)) in zip(ara_scores.items(), eng_scores.items())}}
        
json.dump(model_scores, open(os.path.join("llm_tests", "model_scores.json"), "w"), indent=4)
          
    