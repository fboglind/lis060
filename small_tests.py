"""small_tests.py"""
from collections import Counter
#some_words="these are some words, of which some are reoccurring"
some_words="low lowest newer wider new "

def apply_whitespacing(word):
    """Adds whitespacing and appends an end-of-word symbol to each word."""
    return ' '.join(list(word)) + ' _'

def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

def learn_encoding(words,vocab_size):
    """A method that takes a path to a text corpus (C) and an integer
    (k) indicating the desired vocabulary size(!? not number of merges(?)), and uses the BPE algorithm 
    to compute such a vocabulary"""
    vocab = {char for word in words for char in word if char != ' '} # All the tokens in the text #this is the initial vocab
    print(f"VOCAB: {vocab}")
    k=0
    merged_pairs = set()
    while vocab_size < len(vocab):
        k=+1 
        for i in range(1, k): #from 1 to number of merges k
            #print(f"VOCAB: {vocab}")
            ngram_counts=Counter() #A new counter for each loop
            # now we want to find the most frequent bigram
            # or (bitoken: can be a+r or a+re depending on vocab)
            for word in words:
                ngram_counts(tokens = word.split())  # Split the string by spaces
                

            most_common_ngrams = [ngram for ngram in ngram_counts if ngram not in merged_pairs]
           # print(f"COMMON NGRAMS {most_common_ngrams}")
            merged_pair = max(most_common_ngrams, key=ngram_counts.get)
           # print(f"PAIR: {merged_pair}") 
            vocab.add(merged_pair) # add most freq bigram to vocab
           # print(f"merged_pair[0] in vocab: {merged_pair[0] in vocab}")
           # print(type(vocab))
           # print(type(merged_pair[0]))
        
            for w in words:
                w.replace(merged_pair, "")

        #vocab.discard(merged_pair[0]) #remove first part of bigram from vocab
        #vocab.discard(merged_pair[1]) #remove second part of bigram from vocab
        # We use discard so that no err or will occur if tL==tR
        #Replace each occurence of tL and tR with tNEW (also in text?)
    return vocab

def learn_encoding3(words,k):
    # for i in range(1, k): #from 1 to number of merges k. i is the size of current n-gram
    #     #ngram_size=i
    #     # for w in self.whitespaced_corpus:
    #     for i in range(len(words) - 1):
    #         pass
    return None

def learn_encoding4(words, k):
    for i in range(len(word) - k+1):
        pass 

def learn_encoding5(words, k):
    counts=Counter()
    for pair in zip(ids, ids[1:]):
        counts[pair]

def yielder(s,k):
    """Args:
        s: str
    """
    for i in range(len(s) - k+1): #stop: len(s) + WINDOW - 1 
        yield s[i:i + k]

whitespaced_words_str = " ".join([apply_whitespacing(word) for word in some_words.split()])
whitespaced_words = [apply_whitespacing(word) for word in some_words.split()]

print(f"Original text: {some_words}")
print(f"With whitespacing {whitespaced_words}")

encoding=learn_encoding(whitespaced_words,19)

print(f"Result: {encoding}")
