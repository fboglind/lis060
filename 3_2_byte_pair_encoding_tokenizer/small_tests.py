"""small_tests.py"""
from collections import Counter, defaultdict
#some_words="these are some words, of which some are reoccurring"
some_words="low lowest newer wider new "
with_endtoken="low_ lowest_ newer_ wider_ new_ "#final space?
with_whitespacing="l o w _ l o w e s t _ n e w e r _ w i d e r _ n e w _ "
with_one_merge="l ow _ l ow e s t _ n e w e r _ w i d e r _ n e w _ "
def add_endtoken(word):
    """Appends an end-of-word symbol to each word"""
    return ''.join(list(word)) + '_'

def apply_whitespacing(word, exception=None):
    """Adds whitespacing to each word."""
    return ' '.join(list(word))

# def create_sublists(word):
#     """Creates a list of sublists, where each sublist contains the characters of the word."""
#     return [list(word[i:i + 1]) for i in range(len(word))]

def create_sublists(corpus):
    """Creates a list of sublists, where each sublist contains the characters of each word in the corpus
     while also adding an endtoken to each word."""
    return [[char for char in word + '_'] for word in corpus.split()]

# def compute_pair_freqs(splits):
#     pair_freqs = defaultdict(int)
#     for word, freq in word_freqs.items():
#         split = splits[word]
#         if len(split) == 1:
#             continue
#         for i in range(len(split) - 1):
#             pair = (split[i], split[i + 1])
#             pair_freqs[pair] += freq
#     return pair_freqs

def learn_encoding(corpus, v):
    """A method that takes a path to a text corpus (C) and an integer
    (k) indicating the desired vocabulary size(!? not number of merges(?)), and uses the BPE algorithm 
    to compute such a vocabulary
    * 'The resulting vocabulary consists of the original set of characters plus k new symbols.'
    * desired vocab if k=19: _, d, e, i, l, n, o, r, s, t, w, er, er_,ne, new, lo, low, newer_, low_   
    """
    vocab = {char for word in corpus for char in word if char != ' '} # All the tokens in the text #this is the initial vocab
    initial_vocab_size = len(vocab)
    print(f"INITIAL VOCAB SIZE: {initial_vocab_size}")
    k=v-initial_vocab_size # Number of merges to be performed
    print(f"k={k}")
    merged_pairs = set()
    #current_vocab_size= len(vocab) + len(merged_pairs)
    
    for i in range(1, k): #from 1 to number of merges k    
        ngram_counts=Counter() #A new counter for each loop
        for word in corpus:
            bigrams = yielder(word, 2) # Get all bigrams in the word
            #tokens = word.split()

            ngram_counts.update(bigrams)
            #print(f"TOKENSi:{ngram_counts}")

        
        #common_ngrams = [ngram for ngram in ngram_counts if ngram not in merged_pairs] #These are all(?) new(?) ngrams we encountered 

        # Now we know the most common bigram, it is the max in ngram_counts
        updated_counts=set(ngram_counts.keys()) - merged_pairs # We update the counts by removing the previous merges
        most_common_ngram=max(updated_counts, key=ngram_counts.get) # We calculate a new count
        #print(f"PAIR: {most_common_ngram}") 
        
        vocab.add(most_common_ngram) # add most freq bigram to vocab

        #print(f"new k {i}")
    print(f"FINAL VOCAB SIZE: {len(vocab)}")
    return vocab

def yielder(s,k): #s=string, k= length of ngram
    """A generator that yields all n-grams of length k from the input string s.
    For example, for the string "h e l lo" and k=2, it will (should*) yield "h e", "e l", "l lo".
    Args:
        s: str
    Yields:
        str: n-grams of length k
    """
    for i in range(len(s) - k+1): #stop: len(s) + WINDOW - 1 
        yield s[i:i + k]

def main():
    
    with open("../data/bpe_test.txt", "r") as f:
        corpus = f.read()

#FOR CORPUS (assignment data)
    corpus=" ".join([add_endtoken(word) for word in corpus.split()])
    whitespaced_corpus_str = " ".join([apply_whitespacing(word) for word in corpus.split()])
    
    whitespaced_corpus = [apply_whitespacing(word) for word in corpus.split()]

    sublist_corpus= create_sublists(corpus)

    print(f"ORIGINAL: {corpus}")
    print(type(corpus))
    print(f"WHITESPACED: {whitespaced_corpus}")
    print(type(whitespaced_corpus))
    print(f"SUBLIST: {sublist_corpus}")
    encoding=learn_encoding(whitespaced_corpus,19)
    print(f"RESULT: {encoding}")


#FOR SMALL TEST (a sentence/string)
    
    whitespaced_test = [apply_whitespacing(word) for word in some_words.split()]


    print(f"ORIGINAL (TEST): {some_words}")
    print(type(some_words))
    print(f"WHITESPACED (TEST): {whitespaced_test}")
    
    encoding_test=learn_encoding(whitespaced_test,19)
    print(f"RESULT: {encoding_test}")


if __name__ == "__main__":
    main()


##

# def learn_encoding(words,vocab_size):
#     """A method that takes a path to a text corpus (C) and an integer
#     (k) indicating the desired vocabulary size(!? not number of merges(?)), and uses the BPE algorithm 
#     to compute such a vocabulary"""
#     vocab = {char for word in words for char in word if char != ' '} # All the tokens in the text #this is the initial vocab
#     print(f"VOCAB: {vocab}")
#     k=0
#     merged_pairs = set()
#     while vocab_size < len(vocab):
#         k=+1 
#         for i in range(1, k): #from 1 to number of merges k
#             #print(f"VOCAB: {vocab}")
#             ngram_counts=Counter() #A new counter for each loop
#             # now we want to find the most frequent bigram
#             # or (bitoken: can be a+r or a+re depending on vocab)
#             for word in words:
#                 ngram_counts(tokens = word.split())  # Split the string by spaces
                

#             most_common_ngrams = [ngram for ngram in ngram_counts if ngram not in merged_pairs]
#            # print(f"COMMON NGRAMS {most_common_ngrams}")
#             merged_pair = max(most_common_ngrams, key=ngram_counts.get)
#            # print(f"PAIR: {merged_pair}") 
#             vocab.add(merged_pair) # add most freq bigram to vocab
#            # print(f"merged_pair[0] in vocab: {merged_pair[0] in vocab}")
#            # print(type(vocab))
#            # print(type(merged_pair[0]))
        
#             for w in words:
#                 w.replace(merged_pair, "")

#         #vocab.discard(merged_pair[0]) #remove first part of bigram from vocab
#         #vocab.discard(merged_pair[1]) #remove second part of bigram from vocab
#         # We use discard so that no err or will occur if tL==tR
#         #Replace each occurence of tL and tR with tNEW (also in text?)
#     return vocab
