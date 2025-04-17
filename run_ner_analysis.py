"""Reads conll data and runs program to find Named Entities"""

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

def main():
    train_data = read_conll2003("data/conll2003/eng.train")
    test_data = read_conll2003("data/conll2003/eng.testa")

    # Inspect a sample
    for token, tag in train_data[0]:
        print(f"{token:15} {tag}")


if __name__== "__main__":
    main()


