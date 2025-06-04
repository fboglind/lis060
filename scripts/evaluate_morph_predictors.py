"""evaluate_morpho_models.py
Evaluation script for morphological analysis mini-paper.
Compares Perceptron and MLR on Bulgarian and Russian.
"""
import sys
import os
import csv
import time
import random
from pathlib import Path
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mynltk.morphological_analyzer import MorphologicalAnalyzer
from mynltk.models.perceptron import Perceptron
from mynltk.models.multinomial_regression import MultinomialLogisticRegression as MLR
from mynltk.utils.features import get_features
from mynltk.utils.read_unimorph import read_unimorph
from mynltk.utils.bootstrap_test import paired_bootstrap, accuracy

# Configuration
CONFIG = {
    'data_dir': 'data/unimorph',
    'output_dir': 'results/morpho_comparison',
    'languages': ['bul', 'rus'],
    'models': [('Perceptron', Perceptron), ('MLR', MLR)],
    'n_iters': 5,
    'random_seed': 42
}

def analyze_feature_diversity(train_data):
    """Analyze the diversity of features in training data.
    Returns statistics about unique prefixes, suffixes, and patterns."""
    prefix_counts = defaultdict(set)  # length -> set of unique prefixes
    suffix_counts = defaultdict(set)  # length -> set of unique suffixes
    pos_tags = set()
    unique_forms = set()
    
    for (form, pos), label in train_data:
        form = form.replace(" ", "_")
        pos_tags.add(pos)
        unique_forms.add(form)
        
        for i in range(1, 6):
            if len(form) >= i:
                prefix_counts[i].add(f"{pos} prefix={form[:i]}")
                suffix_counts[i].add(f"{pos} suffix={form[-i:]}")
    
    # Calculate total unique features
    total_prefixes = sum(len(prefixes) for prefixes in prefix_counts.values())
    total_suffixes = sum(len(suffixes) for suffixes in suffix_counts.values())
    
    return {
        'unique_forms': len(unique_forms),
        'pos_tags': len(pos_tags),
        'total_prefixes': total_prefixes,
        'total_suffixes': total_suffixes,
        'total_features': total_prefixes + total_suffixes,
        'prefix_by_length': {i: len(prefix_counts[i]) for i in range(1, 6)},
        'suffix_by_length': {i: len(suffix_counts[i]) for i in range(1, 6)}
    }

def save_predictions(predictions, filepath):
    """Save predictions to file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

def train_and_evaluate(lang, model_name, model_cls, train_data, test_data):
    """Train model and return predictions and metrics."""
    test_words = [(form, pos) for (form, pos), _ in test_data]
    gold_labels = [label for _, label in test_data]
    
    print(f"\nTraining {model_name} on {lang}...")
    start_time = time.time()
    
    analyzer = MorphologicalAnalyzer(
        training_data=train_data,
        n_iters=CONFIG['n_iters'],
        model_cls=model_cls,
        feature_fn=get_features
    )
    
    predictions = analyzer.predict(test_words)
    elapsed_time = time.time() - start_time
    
    acc = accuracy(gold_labels, predictions)
    print(f"Accuracy: {acc:.2%}, Time: {elapsed_time:.1f}s")
    
    return predictions, acc, elapsed_time, analyzer

def main():
    """Run evaluation pipeline."""
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Store results
    results = []
    predictions_cache = {}
    feature_diversity_results = {}
    
    # Process each language
    for lang in CONFIG['languages']:
        train_path = os.path.join(CONFIG['data_dir'], f"{lang}-train")
        test_path = os.path.join(CONFIG['data_dir'], f"{lang}-test")
        
        print(f"\n{'='*50}")
        print(f"Processing {lang.upper()}")
        print(f"{'='*50}")
        
        # Load data (already pre-split and sized appropriately)
        train_data = read_unimorph(train_path)
        test_data = read_unimorph(test_path, testing=False)
        
        print(f"Train: {len(train_data)} examples")
        print(f"Test: {len(test_data)} examples")
        print(f"Labels: {len(set(label for _, label in train_data))} unique")
        
        # Analyze feature diversity
        print(f"\nAnalyzing feature diversity for {lang}...")
        diversity = analyze_feature_diversity(train_data)
        feature_diversity_results[lang] = diversity
        
        print(f"Feature diversity statistics:")
        print(f"  Unique word forms: {diversity['unique_forms']:,}")
        print(f"  POS tags: {diversity['pos_tags']}")
        print(f"  Total unique prefixes: {diversity['total_prefixes']:,}")
        print(f"  Total unique suffixes: {diversity['total_suffixes']:,}")
        print(f"  Total features: {diversity['total_features']:,}")
        
        # Train each model
        for model_name, model_cls in CONFIG['models']:
            predictions, acc, time_taken, analyzer = train_and_evaluate(
                lang, model_name, model_cls, train_data, test_data
            )
            
            # Save predictions
            pred_file = os.path.join(CONFIG['output_dir'], 
                                   f"{lang}_{model_name}_predictions.txt")
            save_predictions(predictions, pred_file)
            
            # Cache for bootstrap
            predictions_cache[(lang, model_name)] = predictions
            
            # Store results
            results.append({
                'language': lang,
                'model': model_name,
                'accuracy': acc,
                'time': time_taken,
                'n_train': len(train_data),
                'n_test': len(test_data),
                'n_labels': len(set(label for _, label in train_data))
            })
    
    # Save main results CSV
    results_file = os.path.join(CONFIG['output_dir'], 'results.csv')
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    # Save feature diversity analysis
    diversity_file = os.path.join(CONFIG['output_dir'], 'feature_diversity.csv')
    with open(diversity_file, 'w', newline='') as f:
        fieldnames = ['language', 'unique_forms', 'pos_tags', 'total_prefixes', 
                     'total_suffixes', 'total_features']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for lang, diversity in feature_diversity_results.items():
            row = {'language': lang}
            row.update({k: v for k, v in diversity.items() 
                       if k in fieldnames})
            writer.writerow(row)
    
    # Save detailed prefix/suffix counts by length
    detailed_file = os.path.join(CONFIG['output_dir'], 'feature_diversity_detailed.txt')
    with open(detailed_file, 'w') as f:
        for lang, diversity in feature_diversity_results.items():
            f.write(f"\n{lang.upper()} Feature Diversity:\n")
            f.write(f"{'='*30}\n")
            f.write(f"Unique word forms: {diversity['unique_forms']:,}\n")
            f.write(f"POS tags: {diversity['pos_tags']}\n")
            f.write(f"Total prefixes: {diversity['total_prefixes']:,}\n")
            f.write(f"Total suffixes: {diversity['total_suffixes']:,}\n")
            f.write(f"\nPrefix counts by length:\n")
            for length, count in diversity['prefix_by_length'].items():
                f.write(f"  Length {length}: {count:,}\n")
            f.write(f"\nSuffix counts by length:\n")
            for length, count in diversity['suffix_by_length'].items():
                f.write(f"  Length {length}: {count:,}\n")
    
    print(f"\nResults saved to:")
    print(f"  - {results_file}")
    print(f"  - {diversity_file}")
    print(f"  - {detailed_file}")
    
    # Return predictions cache for bootstrap script
    return predictions_cache

if __name__ == "__main__":
    main()