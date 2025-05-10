"""naive_bayes_text_classifier.py
A bag-of-words Naive Bayes text classifier based on equations (4.10)
and (4.14) in Jurafsky and Martin (2024, Chapter 4). Test using tokenizer and
CategorizedCorpus classes.
"""
from collections import Counter, defaultdict
import math
from categorized_corpus import CategorizedCorpus
import numpy as np
from tokenizer import Tokenizer


#Loop through each class `c` in `C`:**

#- Count how many documents are in class `c` (`Nc`) 
#- Calculate `logprior[c] = log(Nc / Ndoc)`
#   (i.e., prior probability of class `c`)
#- Collect all documents with label `c` → `bigdoc[c]`


class NaiveBayesTextClassifier:
    """A classifier"""
    def __init__(self) -> None:
        """Initialize the classifier with a corpus"""        
        self.vocab=set()
        self.pos_counts=defaultdict(Counter)
        self.neg_counts=defaultdict(Counter)
        self.pos_size=0
        self.neg_size=0
        self.doc_size=self.pos_size + self.neg_size
        self.train()
        
    def train(self, corpus: CategorizedCorpus):
        """Train the Naive Bayes classifier by building the vocabulary and word counts."""
        for line, sentiment in corpus:
            tokenized_text = Tokenizer.tokenize(line)  # Tokenize the text to get words
            self.vocab.update(tokenized_text)  # Add words to vocabulary
            if sentiment == "pos":  # Check if sentiment is positive
                self.pos_counts.update(tokenized_text)  # Update the positive word counts
                self.pos_size=+1
            else:  # Otherwise, update the negative word counts
                self.neg_counts.update(tokenized_text)
                self.neg_size=+1



    @staticmethod
    def predict_corpus(corpus:CategorizedCorpus):
        """Predicts all sentences"""
        for line, _ in corpus:
            sentiment=None
            for word in line:
                pos_sum=sum(corpus.pos_counts.get(word, 0) for word in line)
                neg_sum=sum(corpus.neg_counts.get(word, 0) for word in line)
                if pos_sum>neg_sum:
                    sentiment = 1
                elif neg_sum>pos_sum:
                    sentiment = 0

    @staticmethod
    def predict(text:list[str]):
        """Predicts one sentece"""
        for word in text:
            



    @staticmethod
    def evaluate():
        pass
        
def main() -> None:
    """Main function"""
    imdb_train = CategorizedCorpus("data/imdb/train")
    imbd_test = CategorizedCorpus("data/imdb/test") 

if __name__=="__main__":
    main()