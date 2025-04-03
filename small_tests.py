"""small_tests.py"""
from collections import Counter
some_words="these are some words, of which some are reoccurring"

def apply_whitespacing(word):
    """Adds whitespacing and appends an end-of-word symbol to each word."""
    return ''.join(list(word)) + '_'

def learn_encoding(words,k):
    """A method that takes a path to a text corpus (C) and an integer
    (k) indicating the desired vocabulary size, and uses the BPE algorithm 
    to compute such a vocabulary"""
    vocab = {char for word in words for char in word} # V
    n_gram_counts=Counter()

    for i in range(1, k): #from 1 to number of merges k. i is the size of current n-gram
        for w in words:
            print(w[i:i+2])

    
    
    
    # for i in range(1, k): #from 1 to number of merges k. i is the size of current n-gram
    #     #ngram_size=i
    #     # for w in self.whitespaced_corpus:
    #     for i in range(len(words) - 1):
    #         pass
    
    return None


whitespaced_words_str = " ".join([apply_whitespacing(word) for word in some_words.split()])
whitespaced_words = [apply_whitespacing(word) for word in some_words.split()]


print(whitespaced_words)

encoding=learn_encoding(whitespaced_words,5)
#[i:i+2]