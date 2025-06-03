#!/usr/bin/env python3
"""
count_multiword_expressions.py
Counts and analyzes multiword expressions in UniMorph datasets
"""

import os
from collections import defaultdict

def analyze_multiword_expressions(filepath):
    """
    Analyze multiword expressions in a UniMorph file
    
    Args:
        filepath: Path to the UniMorph file
        
    Returns:
        dict: Statistics about multiword expressions
    """
    stats = {
        'total_lines': 0,
        'valid_lines': 0,
        'space_separated': 0,
        'underscore_separated': 0,
        'both_separators': 0,
        'single_word': 0,
        'multiword_examples': [],
        'space_examples': [],
        'underscore_examples': []
    }
    
    print(f"\nAnalyzing: {filepath}")
    print("-" * 60)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                stats['total_lines'] += 1
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse the line
                parts = line.split('\t')
                if len(parts) != 3:
                    continue
                
                stats['valid_lines'] += 1
                lemma, form, tags = parts
                
                # Check for multiword expressions in the inflected form
                has_space = ' ' in form
                has_underscore = '_' in form
                
                if has_space and has_underscore:
                    stats['both_separators'] += 1
                    if len(stats['multiword_examples']) < 10:
                        stats['multiword_examples'].append((form, tags))
                elif has_space:
                    stats['space_separated'] += 1
                    if len(stats['space_examples']) < 10:
                        stats['space_examples'].append((form, tags))
                elif has_underscore:
                    stats['underscore_separated'] += 1
                    if len(stats['underscore_examples']) < 10:
                        stats['underscore_examples'].append((form, tags))
                else:
                    stats['single_word'] += 1
    
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    # Calculate percentages
    if stats['valid_lines'] > 0:
        stats['space_percentage'] = (stats['space_separated'] / stats['valid_lines']) * 100
        stats['underscore_percentage'] = (stats['underscore_separated'] / stats['valid_lines']) * 100
        stats['both_percentage'] = (stats['both_separators'] / stats['valid_lines']) * 100
        stats['single_percentage'] = (stats['single_word'] / stats['valid_lines']) * 100
        stats['total_multiword'] = stats['space_separated'] + stats['underscore_separated'] + stats['both_separators']
        stats['multiword_percentage'] = (stats['total_multiword'] / stats['valid_lines']) * 100
    
    return stats

def print_stats(stats, language):
    """Print statistics in a readable format"""
    if stats is None:
        return
    
    print(f"\n=== STATISTICS FOR {language.upper()} ===")
    print(f"Total lines processed: {stats['total_lines']}")
    print(f"Valid data lines: {stats['valid_lines']}")
    print(f"Single words: {stats['single_word']} ({stats['single_percentage']:.1f}%)")
    print(f"Space-separated: {stats['space_separated']} ({stats['space_percentage']:.1f}%)")
    print(f"Underscore-separated: {stats['underscore_separated']} ({stats['underscore_percentage']:.1f}%)")
    print(f"Both separators: {stats['both_separators']} ({stats['both_percentage']:.1f}%)")
    print(f"TOTAL MULTIWORD: {stats['total_multiword']} ({stats['multiword_percentage']:.1f}%)")
    
    # Show examples
    if stats['space_examples']:
        print(f"\nSpace-separated examples:")
        for form, tags in stats['space_examples'][:5]:
            print(f"  '{form}' -> {tags}")
    
    if stats['underscore_examples']:
        print(f"\nUnderscore-separated examples:")
        for form, tags in stats['underscore_examples'][:5]:
            print(f"  '{form}' -> {tags}")
    
    if stats['multiword_examples']:
        print(f"\nBoth separators examples:")
        for form, tags in stats['multiword_examples'][:5]:
            print(f"  '{form}' -> {tags}")

def main():
    """Main function to analyze multiple language datasets"""
    data_dir = "data/unimorph"
    languages = ["swe", "rus", "bul"]
    datasets = ["train", "test"]
    
    print("MULTIWORD EXPRESSION ANALYSIS")
    print("=" * 60)
    
    summary = {}
    
    for lang in languages:
        summary[lang] = {}
        for dataset in datasets:
            filepath = os.path.join(data_dir, f"{lang}-{dataset}")
            if os.path.exists(filepath):
                stats = analyze_multiword_expressions(filepath)
                print_stats(stats, f"{lang}-{dataset}")
                summary[lang][dataset] = stats
            else:
                print(f"\nFile not found: {filepath}")
    
    # Print summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Language':<10} {'Dataset':<8} {'Total':<8} {'Multiword':<10} {'Percentage':<10}")
    print("-" * 60)
    
    for lang in languages:
        for dataset in datasets:
            if lang in summary and dataset in summary[lang] and summary[lang][dataset]:
                stats = summary[lang][dataset]
                print(f"{lang:<10} {dataset:<8} {stats['valid_lines']:<8} {stats['total_multiword']:<10} {stats['multiword_percentage']:<10.1f}%")

if __name__ == "__main__":
    main()