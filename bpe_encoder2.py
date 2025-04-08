"""bpe_encoder2.py"""
from collections import Counter
from tokenizer import Tokenizer
from morphological_analyzer import MorphologicalAnalyzer

class BPEncoder:
    """ Preprocessing:
    The algorithm is usually run inside words (not merging across word boundaries),
    so the input corpus is first white-space-separated to give a set of strings, each
    corresponding to the characters of a word, plus a special end-of-word symbol"""
    
    def __init__(self, corpus, k, use_true_bytes=False):
        self.corpus=corpus # C
        self.processed_corpus = [self.apply_whitespacing(word) for word in corpus.split()] # C
        self.vocab={char for word in self.processed_corpus for char in word} # V
        self.k=k # k
        self.use_true_bytes=use_true_bytes
        self.encoded_corpus = [self.learn_encoding(word) for word in self.corpus] # C'
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

    def learn_encoding(self)-> None:
        """A method that takes a path to a text corpus (C) and an integer
        (k) indicating the desired vocabulary size, and uses the BPE algorithm
        to compute such a vocabulary"""
        
        # ngram_counts=Counter()
        # for i in range(self.k): # set ngram size
        #     for i in range(len(self.preprocessed_text) - 1): 
        #         ngram = self.preprocessed_text[i:i+2]
        #         ngram_counts[ngram] += 1   
    
    def learn_encoding2(self)->dict:
            counts = {}
            for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
                counts[pair] = counts.get(pair, 0) + 1
            return counts
                
    
    def apply_encoding(self, text:str)->str:
        """A method that reads a text file and produces a tokenized text, using
        the BPE vocabulary produced by the previous method."""
        pass
    def load_encoding(self):
        """ Functionality to load the BPE vocabulary from a file, in order
        to avoid having to run the (slow) BPE algorithm more than once. """
        pass

    def save_encoding(self, filname:str):
        """ Functionality to save the BPE vocabulary to a file, in order
        to avoid having to run the (slow) BPE algorithm more than once. """
        pass
