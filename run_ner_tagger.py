"""Run NER tagger on CoNLL2003 dataset

This script trains and evaluates a structured perceptron for
Named Entity Recognition on the CoNLL2003 dataset.
"""
from ner_tagger import NERTagger


def read_conll2003(filepath: str) -> list[tuple[list[str], list[str]]]:
    """Reads CoNLL-2003 format data and returns a list of (words, tags) tuples.
    Each tuple contains a list of words and a list of corresponding tags.
    Args:
        filepath: Path to the CoNLL-2003 formatted file.
    Returns:
        A list of tuples, where each tuple contains a list of words and a list of tags.
    """
    data = []
    current_words = []
    current_tags = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                # End of sentence
                if current_words:
                    data.append((current_words, current_tags))
                    current_words = []
                    current_tags = []
                continue
            if line.startswith("-DOCSTART-"):
                continue
            parts = line.split()
            if len(parts) != 4:
                continue  # Skip malformed lines
            word, pos, chunk, ner_tag = parts
            current_words.append(word)
            current_tags.append(ner_tag)

    # Append last sentence if file doesn't end with a newline
    if current_words:
        data.append((current_words, current_tags))

    return data


def main():
    """Main function to run the NER tagger."""
    # Read data directly in the format needed for the perceptron
    print("Reading training data...")
    train_data = read_conll2003("../data/conll2003/eng.train")
    print(f"Read {len(train_data)} training sequences")
    
    print("Reading test data...")
    test_data = read_conll2003("../data/conll2003/eng.testa")
    print(f"Read {len(test_data)} test sequences")
    
    # Create and train NER tagger
    print("Training NER tagger...")
    tagger = NERTagger(n_iters=3)  # Using 3 epochs as specified in assignment
    tagger.train(train_data)
    
    # Evaluate on test data
    print("Evaluating on test data...")
    results = tagger.evaluate(test_data)
    
    print("\nEvaluation Results:")
    print(f"Token Accuracy: {results['token_accuracy']:.4f}")
    print(f"Sequence Accuracy: {results['sequence_accuracy']:.4f}")
    
    # Show top features
    print("\nTop Features:")
    tagger.show_top_features(n=10)
    
    # Analyze a sample sentence
    sample_idx = 2  # Change this index to analyze different samples
    sample_words, sample_gold_tags = test_data[sample_idx]
    sample_pred_tags = tagger.predict([sample_words])[0]
    
    print("\nSample Analysis:")
    print(f"{'Word':<15} {'Gold Tag':<10} {'Predicted Tag':<15}") #Formatting output
    print("-" * 40)
    for word, gold, pred in zip(sample_words, sample_gold_tags, sample_pred_tags):
        print(f"{word:<15} {gold:<10} {pred:<15}")


if __name__ == "__main__":
    main()
