"""run_ner_tagger - runs NER tagger on CoNLL2003 dataset

This script trains and evaluates a structured perceptron for
Named Entity Recognition on the CoNLL2003 dataset.
"""
from structured_perceptron import NERTagger


def read_conll2003(filepath: str) -> list[list[tuple[str, str]]]:
    """
    Reads CoNLL-2003 format data and returns a list of sentences.
    Each sentence is a list of (word, tag) pairs.
    """
    sequences = []
    current_seq = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_seq:
                    sequences.append(current_seq)
                    current_seq = []
                continue
            if line.startswith("-DOCSTART-"):
                continue
            parts = line.split()
            if len(parts) != 4:
                continue  # Skip malformed lines
            word, pos, chunk, ner_tag = parts
            current_seq.append((word, ner_tag))

    # Append last sentence if file doesn't end with a newline
    if current_seq:
        sequences.append(current_seq)

    return sequences


def prepare_data(sequences):
    """
    Convert CoNLL format sequences to format used by the structured perceptron.
    Returns list of (words, tags) tuples.
    """
    data = []
    for sequence in sequences:
        words = [token for token, _ in sequence]
        tags = [tag for _, tag in sequence]
        data.append((words, tags))
    return data


def main():

    print("Reading training data...")
    train_sequences = read_conll2003("data/conll2003/eng.train")
    print(f"Read {len(train_sequences)} training sequences")
    
    print("Reading test data...")
    test_sequences = read_conll2003("data/conll2003/eng.testa")
    print(f"Read {len(test_sequences)} test sequences")
    
    # Prepare data for structured perceptron
    train_data = prepare_data(train_sequences)
    test_data = prepare_data(test_sequences)
    
    print(train_sequences[0])
    print(train_data[0])

if __name__ == "__main__":
    main()
