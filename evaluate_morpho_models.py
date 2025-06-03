"""evaluate_morpho_models.py
Runs both Perceptron and Multinomial Logistic Regression (MLR) classifiers on
Russian and Bulgarian UniMorph datasets, computes accuracy, and saves results to CSV.
"""

import os
from mynltk.morphological_analyzer import MorphologicalAnalyzer
from mynltk.models.perceptron import Perceptron
from mynltk.multinomial_regression import MultinomialLogisticRegression as MLR
from mynltk.utils.features import get_features
from random import shuffle

DATA_DIR = "data/unimorph"
RESULTS_CSV = "morpho_eval_results.csv"
LANGUAGES = ["rus", "bul"]
MODELS = [("Perceptron", Perceptron), ("MLR", MLR)]

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

        for model_name, model_cls in MODELS:
            print(f"Training {model_name} on {lang}...")

            analyzer = MorphologicalAnalyzer(train_data, n_iters=5, model_cls=model_cls, feature_fn=get_features)
            predictions = analyzer.predict(test_words)
            accuracy = compute_accuracy(test_data, predictions)
            print(f"{model_name} on {lang}: Accuracy = {accuracy:.2%}")
            rows.append((lang, model_name, f"{accuracy:.4f}"))

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Saved results to {RESULTS_CSV}")

if __name__ == "__main__":
    main()
