from mynltk import BPEncoder


def main():
    """Main function to demonstrate the BPE encoding process."""
    # Read data
    filepath = "data/bpe_test.txt"
    # Create a corpus
    corpus = BPEncoder.read_corpus(filepath)
    # Create an encoder
    encoder = BPEncoder(corpus, vocab_size=19)
    # Apply encoding
    tokenized_text = encoder.apply_encoding(filepath)
    # Save encoding
    encoder.save_encoding("bpe_model.json")
    
    # Print results
    print(f"ENCODING: {encoder.encoding}")
    print(f"MERGED PAIRS: {encoder.merges}")
    print(f"INITIAL VOCAB SIZE: {len(encoder.initial_vocab)}, k={encoder.k},FINAL VOCAB SIZE: {len(encoder.encoding)}\n")

    print("PROCESSED CORPUS:")
    print(tokenized_text)
   
    # # Load encoder from file
    # new_encoder = BPEncoder(corpus="", vocab_size=0)  # empty
    # new_encoder.load_encoding("bpe_model.json")
    # print(f"ENCODING: {new_encoder.encoding}")
    # print(f"MERGED PAIRS: {new_encoder.merges}")
    # print(f"INITIAL VOCAB SIZE: {len(new_encoder.initial_vocab)}, k={new_encoder.k},FINAL VOCAB SIZE: {len(new_encoder.encoding)}\n")

    # print("PROCESSED CORPUS:")
    # print(tokenized_text)
  
if __name__ == "__main__":
    main()
