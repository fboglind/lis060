"""
Run bootstrap significance tests for morphological analysis comparison.
Should be run after evaluate_morpho_models.py
"""
import sys
import os
import csv
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mynltk.utils.read_unimorph import read_unimorph
from mynltk.utils.bootstrap_test import paired_bootstrap

def load_predictions(filepath):
    """Load predictions from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def main():
    results_dir = 'results/morpho_comparison'
    
    print("Running Bootstrap Significance Tests")
    print("="*50)
    
    # Test model differences within each language
    for lang in ['bul', 'rus']:
        print(f"\n{lang.upper()}: Comparing MLR vs Perceptron")
        
        # Load gold labels
        test_path = f'data/unimorph/{lang}-test'
        test_data = read_unimorph(test_path, testing=False)
        gold_labels = [label for _, label in test_data]
        
        # Load predictions
        perc_pred = load_predictions(f'{results_dir}/{lang}_Perceptron_predictions.txt')
        mlr_pred = load_predictions(f'{results_dir}/{lang}_MLR_predictions.txt')
        
        # Run bootstrap test
        delta, p_value = paired_bootstrap(gold_labels, mlr_pred, perc_pred)
        
        print(f"  Δ (MLR - Perceptron): {delta:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")
    
    print("\nBootstrap testing complete!")

if __name__ == "__main__":
    main()