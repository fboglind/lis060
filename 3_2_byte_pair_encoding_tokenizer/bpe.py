"""bpe.py"""
from collections import Counter, defaultdict
import json
import os

    #while size of merges < vocab_size:
    #count frequencies of adjacent symbol pairs
    #find most frequent pair
    #merge the pair throughout the corpus
    #add the pair to the merges list

some_words="low lowest newer wider new "

class BPEncoder:
    def __init__(self, corpus: str, vocab_size: int, use_true_bytes:bool=False):
        self.processed_corpus=self.add_endtoken(corpus)
        self.whitespaced_corpus=self.apply_whitespacing(self.processed_corpus)
        self.vocab = {char for word in self.processed_corpus for char in word if char != ' '}
        self.k=vocab_size-len(self.vocab) #number of merges to perform
        self.encoding=self.learn_encoding()

    def learn_encoding(self):
        """A method that takes a path to a text corpus (C) and an integer
        (k) indicating the desired vocabulary size(!? not number of merges(?)), and uses the BPE algorithm 
        to compute such a vocabulary
        * 'The resulting vocabulary consists of the original set of characters plus k new symbols.'
        * desired vocab if k=19: _, d, e, i, l, n, o, r, s, t, w, er, er_,ne, new, lo, low, newer_, low_   
        """
        
        print(f"INITIAL VOCAB SIZE: {len(self.vocab)}")
        print(f"k={self.k}")
        merged_pairs = set()

        for i in range(1, self.k + 1):  # Perform k merges
            ngram_counts = Counter()  # A new counter for each loop
            for word in self.processed_corpus.split():
                bigrams = self.yielder(word, 2)  # Get all bigrams in the word
                ngram_counts.update(bigrams)
#print(f"TOKENSi:{ngram_counts}")

            
            #common_ngrams = [ngram for ngram in ngram_counts if ngram not in merged_pairs] #These are all(?) new(?) ngrams we encountered 

            # Now we know the most common bigram, it is the max in ngram_counts
            updated_counts=set(ngram_counts.keys()) - merged_pairs # We update the counts by removing the previous merges
            most_common_ngram=max(updated_counts, key=ngram_counts.get) # We calculate a new count
            print(f"PAIR: {most_common_ngram}")

            self.vocab.add(most_common_ngram) # add most freq bigram to vocab
            
            print(f"new k {i}")

        print(f"FINAL VOCAB SIZE: {len(self.vocab)}")
        return self.vocab  # Return the updated vocabulary

    def yielder(self,s,k): #s=string, k= length of ngram
        """A generator that yields all n-grams of length k from the input string s.
        For example, for the string "h e l lo" and k=2, it will (should*) yield "h e", "e l", "l lo".
        Args:
            s: str
        Yields:
            str: n-grams of length k
        """
        for i in range(len(s) - k+1): #stop: len(s) + WINDOW - 1 
            yield s[i:i + k]


    def add_endtoken(self, corpus:str) ->str:
        """Appends an end-of-word symbol to each word in a corpus"""
        return ' '.join([word + '_' for word in corpus.split()])

    def apply_whitespacing(self, corpus:str -> str, exception=None):
        """Adds whitespacing to a words in a corpus.
        """
        return ' '.join(corpus.replace(' ', ''))


def main():
    
    with open("../data/bpe_test.txt", "r") as f:
        corpus = f.read()
    
    encoder = BPEncoder(some_words, vocab_size=19)
    print(f"ENDTOKENED: {encoder.processed_corpus}")
    print(f"WHITESPACED: {encoder.whitespaced_corpus}")
    print(f"ENCODING: {encoder.encoding}")

if __name__ == "__main__":
    main()


