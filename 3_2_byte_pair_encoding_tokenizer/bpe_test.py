from bpencoder2 import BPEncoder

def main():

    with open("../data/bpe_test.txt", "r") as f:
        corpus = f.read()

    encoder = BPEncoder(corpus, k=19)
    tokenized_text = encoder.apply_encoding(corpus)
    print(f"CORPUS: {corpus}")
    print(f"TOKENS: {tokenized_text}")

    encoder.save_encoding("bpe_vocab.json")

    # encoder.load_encoding("bpe_vocab.json")
    # tokenized = encoder.apply_encoding("new words here")


if __name__ == "__main__":
    main()
