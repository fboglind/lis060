"""bpencoder2.py"""
from collections import Counter, defaultdict
import json
import os

class BPEncoder:
    def __init__(self, corpus, k, use_true_bytes=False):
        self.corpus = corpus
        self.k = k
        self.use_true_bytes = use_true_bytes
        self.vocab = set()
        self.bpe_vocab = {}
        self.whitespaced_corpus = [self.apply_whitespacing(word) for word in corpus.split()]
        self.processed_corpus = self.whitespaced_corpus if not self.use_true_bytes else [
            self.apply_byte_conversion(word) for word in corpus.split()
        ]
        self.learn_encoding(self.processed_corpus, k)

    def apply_whitespacing(self, word):
        """Adds whitespacing and appends an end-of-word symbol to each word."""
        return ' '.join(list(word)) + ' _'
    
    def merge_pair(self, pair, corpus):
        """Merges the most frequent pair in the corpus."""
        first, second = pair
        new_corpus = []
        for word in corpus:
            new_word = word.replace(' '.join(pair), ''.join(pair))
            new_corpus.append(new_word)
        return new_corpus

    def apply_byte_conversion(self, word):
        """Converts a word to its byte representation."""
        tokens = word.encode('utf-8')
        return ' '.join(map(str, tokens)) + ' _'

    def get_stats(self, corpus):
        pairs = Counter()
        for word in corpus:
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += 1
        return pairs

    def learn_encoding(self, words, k):
        """A method that takes a path to a text corpus (C) and an integer
        (k) indicating the desired vocabulary size, and uses the BPE algorithm 
        to compute such a vocabulary"""
        vocab=defaultdict(int)
        vocab = defaultdict(int)
        corpus = words[:]

        for i in range(k):
            pairs = self.get_stats(corpus)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.bpe_vocab[best] = ''.join(best)
            corpus = self.merge_pair(best, corpus) # 
        self.processed_corpus = corpus

    def apply_encoding(self, text: str) -> str:
        words = [self.apply_whitespacing(word) for word in text.split()]
        for pair, merged in self.bpe_vocab.items():
            bigram = ' '.join(pair)
            words = [word.replace(bigram, merged) for word in words]
        return '\n'.join(words)

    def save_encoding(self, filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({f"{k[0]} {k[1]}": v for k, v in self.bpe_vocab.items()}, f)

    def load_encoding(self, filename: str):
        with open(filename, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        self.bpe_vocab = {tuple(k.split()): v for k, v in loaded.items()}
