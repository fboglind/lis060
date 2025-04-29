"""bpe.py"""
from collections import Counter
from typing import Generator
import json

class BPEncoder:
    def __init__(self, corpus: str, vocab_size: int):
        self.processed_corpus=self.preprocess_corpus(corpus)
        self.vocab = {char for line in self.processed_corpus for word in line for char in word if char != ' '}
        self.initial_vocab = frozenset(self.vocab.copy())
        self.k = max(0, vocab_size - len(self.initial_vocab)) #-2 # For producing expected output
        self.merges = []       
        self.encoding=self.learn_encoding()


    def learn_encoding(self)-> set[str]:
        """A method that uses the BPE algorithm to computes a vocabulary of a text
        consisting of the original set of characters plus k new symbols obtained by merging
        the most frequent pairs of symbols in the text.  
        """
        
        print(f"k={self.k}")
        merged_pairs = set()

        for _ in range(1, self.k +1):  # Perform k merges
            ngram_counts = Counter()  # A new counter for each loop
            for line in self.processed_corpus:
                for word in line: # the corpus is a list...
                    bigrams = self.yielder(tuple(word), 2)  # Convert word to tuple and get all bigrams
                    ngram_counts.update(bigrams) 
            
            # Now we know the most common bigram, it is the max in ngram_counts
            updated_counts=set(ngram_counts.keys()) - merged_pairs # We update the counts by removing the previous merges
            most_common=max(updated_counts, key=ngram_counts.get) # We calculate a new count
            self.merges.append(most_common)
            self.vocab.add(''.join(most_common)) # add most freq bigram to vocab
            
            # Finally, we update the corpus:
            self.merge_pair(self.processed_corpus, most_common)

        return self.vocab  # We return the updated vocabulary
    
    def merge_pair(self, text: list[list[list[str]]], pair: tuple[str, str])-> None:
        """
        Given a text (list of lines -> words -> symbols),
        merges the specified pair everywhere it occurs.
        """
        for line_idx, line in enumerate(text):
            for word_idx, word in enumerate(line):
                merged_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                        merged_word.append(word[i] + word[i+1])
                        i += 2
                    else:
                        merged_word.append(word[i])
                        i += 1
                text[line_idx][word_idx] = merged_word


    def yielder(self, word: tuple[str, str], k: int) -> Generator[tuple[str, ...], None, None]:
        """A generator that yields all n-grams of length k from the input word.
        For example, for the string "h e l lo" and k=2, it will (should*) yield "h e", "e l", "l lo".
        Args:
            word: tuple[str, str]
            k: int
        Yields:
            tuple[str, ...]: n-grams of length k
        """
        for i in range(len(word) - k + 1):  # stop: len(word) + WINDOW - 1
            yield word[i:i + k]

    def preprocess_corpus(self, corpus: str) -> list[list[list[str]]]:
        """Preprocesses the corpus by creating sublists while preserving linebreaks."""
        return [
            [[char for char in word + '_'] for word in line.split()]
            for line in corpus.strip().split('\n')
        ]

    def apply_encoding(self, filepath: str) -> str:
        """Reads a text file and tokenizes it using the learned BPE vocabulary."""
        corpus = self.read_corpus(filepath)
        
        # Preprocess: split corpus into lines, words, characters
        text = self.preprocess_corpus(corpus)
        # Apply BPE merges
        for pair in self.merges:
            self.merge_pair(text, pair)

        # Build the final tokenized text string
        return '\n'.join([' '.join(' '.join(symbol for symbol in word) for word in line) for line in text])
         
    def save_encoding(self, filepath: str)-> None:
        """Save the BPE encoding and vocab size to a JSON file."""
        state = {
            "vocab_size": len(self.initial_vocab) + len(self.merges),
            "merges": list(self.merges)
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_encoding(self, filepath: str)-> None:
        """Load the BPE encoding and vocab size from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.merges = [tuple(pair) for pair in state["merges"]]
        self.k = state["vocab_size"] - len(self.initial_vocab) # restore k


    @staticmethod                  
    def read_corpus(filepath):
        with open(filepath, "r") as f:
            corpus = f.read()
        return corpus


def main():
    """Main function to demonstrate the BPE encoding process."""
    # Read data
    filepath = "../data/bpe_test.txt"
    # Create a corpus
    corpus = BPEncoder.read_corpus(filepath)
    # Create an encoder
    encoder = BPEncoder(corpus, vocab_size=19)
    # Apply encoding
    tokenized_text = encoder.apply_encoding(filepath)
    #Save encoding
    encoder.save_encoding("bpe_model.json")
    
    # Print results
    print(f"ENCODING: {encoder.encoding}")
    print(f"MERGED PAIRS: {encoder.merges}")
    print(f"INITIAL VOCAB SIZE: {len(encoder.initial_vocab)}, k={encoder.k},FINAL VOCAB SIZE: {len(encoder.encoding)}\n")

    print("PROCESSED CORPUS:")
    print(tokenized_text)
    #print(f"MERGES: {encoder.merges}, Nr OF MERGES: {len(encoder.merges)}")

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
