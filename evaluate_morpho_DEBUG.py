"""evaluate_morpho_models.py - DEBUG VERSION
Runs both Perceptron and Multinomial Logistic Regression (MLR) classifiers on
Russian and Bulgarian UniMorph datasets, computes accuracy, and saves results to CSV.
"""

import os
from mynltk.morphological_analyzer import MorphologicalAnalyzer
from mynltk.models.perceptron import Perceptron
from mynltk.multinomial_regression import MultinomialLogisticRegression as MLR
from mynltk.utils.features import get_features
from random import shuffle
import time

DATA_DIR = "data/unimorph"
RESULTS_CSV = "morpho_eval_results.csv"
LANGUAGES = ["rus"]  # Start with just Russian for debugging
#MODELS = [("Perceptron", Perceptron)]  # Start with just Perceptron
MODELS = [MLR,]
def load_data(file_path):
    with open(file_path, encoding="utf-8") as f:
        data = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            try:
                lemma, form, tag = line.split("\t")
                form = form.replace(" ", "_")
                pos = tag.split(";")[0]
                data.append(((form, pos), tag))
            except ValueError:
                continue
        return data

def compute_accuracy(gold_data, predicted_tags):
    correct = sum(1 for ((_, _), gold), pred in zip(gold_data, predicted_tags) if gold == pred)
    return correct / len(gold_data)

def main():
    rows = [("Language", "Model", "Accuracy")]

    for lang in LANGUAGES:
        train_file = os.path.join(DATA_DIR, f"{lang}-train")
        test_file = os.path.join(DATA_DIR, f"{lang}-test")

        print(f"Loading {lang}...")
        train_data = load_data(train_file)
        test_data = load_data(test_file)
        test_words = [x for x, _ in test_data]
        
        print(f"Train data size: {len(train_data)}")
        print(f"Test data size: {len(test_data)}")
        print(f"Sample train data: {train_data[:3]}")
        print(f"Sample test words: {test_words[:3]}")
        print(f"Unique labels: {len(set(label for _, label in train_data))}")

        for model_cls in MODELS:
            model_name = model_cls.__name__
            print(f"Training {model_name} on {lang}...")
            start_time = time.time()

            try:
                analyzer = MorphologicalAnalyzer(train_data, n_iters=3, model_cls=model_cls, feature_fn=get_features)
                print("Analyzer created, starting predictions...")
                
                # Try just one prediction first
                print("Testing single prediction...")
                single_prediction = analyzer.predict([test_words[0]])
                print(f"Single prediction successful: {single_prediction}")
                
                # If that works, try a small batch
                print("Testing small batch...")
                small_batch = analyzer.predict(test_words[:10])
                print(f"Small batch successful: {len(small_batch)} predictions")
                
                # If that works, do the full prediction
                print("Starting full prediction...")
                predictions = analyzer.predict(test_words)
                
                accuracy = compute_accuracy(test_data, predictions)
                elapsed = time.time() - start_time
                print(f"{model_name} on {lang}: Accuracy = {accuracy:.2%} (took {elapsed:.1f}s)")
                rows.append((lang, model_name, f"{accuracy:.4f}"))
                
            except Exception as e:
                print(f"Error with {model_name} on {lang}: {e}")
                import traceback
                traceback.print_exc()

    print("Done!")

if __name__ == "__main__":
    main()