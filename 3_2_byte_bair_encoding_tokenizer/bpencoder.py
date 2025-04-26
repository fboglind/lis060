"""bpencoder.py"""
from collections import Counter, defaultdict
import json
import os
#from tokenizer import Tokenizer

class BPEncoder:
    """ Preprocessing:
    The algorithm is usually run inside words (not merging across word boundaries),
    so the input corpus is first white-space-separated to give a set of strings, each corre
    sponding to the characters of a word, plus a special end-of-word symbol"""
    
    def __init__(self, corpus, k, use_true_bytes=False):
        self.corpus=corpus # C
        self.k=k # k
        #self.use_true_bytes=use_true_bytes
        self.whitespaced_corpus = [self.apply_whitespacing(word) for word in corpus.split()] # C
        #self.vocab={char for word in self.processed_corpus for char in word} # V
        self.learn_encoding(self.processed_corpus, k)
        
        #self.encode=self.learn_encoding() # function
        

    def apply_whitespacing(self, word):
        """Adds whitespacing and appends an end-of-word symbol to each word."""
        return ' '.join(list(word)) + ' _'
    
    def apply_byte_conversion(self, word):
        """Converts a string to a list of integer tokens."""
        tokens = word.encode('utf-8') # raw bytes
        tokens = list(map(int, tokens))
        return tokens
        #return [char.encode('utf-8') for char in word]

    def get_stats(self, corpus):
        pairs = Counter()
        for word in corpus:
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += 1
        return pairs

    def learn_encoding(self, words,k):
        """A method that takes a path to a text corpus (C) and an integer
        (k) indicating the desired vocabulary size, and uses the BPE algorithm 
        to compute such a vocabulary"""
        vocab=defaultdict(int)
        corpus=[]
        ngram_counts=Counter()
        merged_pairs = set()
        for i in range(1, k): #from 1 to number of merges k. i is the size of current n-gram
            vocab = {char for word in words for char in word} # All the tokens in the text
            print(f"VOCAB: {vocab}")
            ngram_counts.clear()
            for w in words: # for each letter
                if i < len(w):
                    ngram_counts[w[i:i+2]] =+1 #actually, for each bigram
            most_common_ngrams = [ngram for ngram in ngram_counts if ngram not in merged_pairs]
            merged_pair = max(most_common_ngrams, key=ngram_counts.get)
            print(f"PAIR: {merged_pair}")
            
            vocab.add(merged_pair) # add most freq bigram to vocab
            print(f"merged_pair[0] in vocab: {merged_pair[0] in vocab}")
            print(type(vocab))
            print(type(merged_pair[0]))
            
            for w in words:
                w.replace(merged_pair, "")

            #vocab.discard(merged_pair[0]) #remove first part of bigram from vocab
            #vocab.discard(merged_pair[1]) #remove second part of bigram from vocab
            # We use discard so that no err or will occur if tL==tR
            #Replace each occurence of tL and tR with tNEW (also in text?)
        return vocab   

    
    def apply_encoding(self, text:str)->str:
        """A method that reads a text file and produces a tokenized text, using
        the BPE vocabulary produced by the previous method."""
        pass

    def save_encoding(self, filename: str):
        """ Functionality to save the BPE vocabulary to a file, in order
        to avoid having to run the (slow) BPE algorithm more than once. """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({f"{k[0]} {k[1]}": v for k, v in self.bpe_vocab.items()}, f)

    def load_encoding(self, filename: str):
        """ Functionality to load the BPE vocabulary from a file, in order
        to avoid having to run the (slow) BPE algorithm more than once. """
        with open(filename, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        self.bpe_vocab = {tuple(k.split()): v for k, v in loaded.items()}