"""naive_bayes_text_classifier.py

A bag-of-words Naive Bayes text classifier based on equations (4.10)
and (4.14) in Jurafsky and Martin (2024, Chapter 4). Test using tokenizer and
CategorizedCorpus classes.
"""
from collections import Counter
import math
from mynltk import CategorizedCorpus
from mynltk import Tokenizer


class NaiveBayesTextClassifier:
    """A classifier"""
    def __init__(self) -> None:
        """Initialize the classifier with a corpus"""        
        self.tokenizer = Tokenizer()  # Create an instance of the Tokenizer class
        self.vocab=set()
        self.pos_counts = Counter()
        self.neg_counts = Counter()
        self.doc_size=0
        self.pos_size=0
        self.neg_size=0
        self.pos_logprior = None
        self.neg_logprior = None
        self.loglikelihood = {}
        
        
 
    def train(self, corpus: CategorizedCorpus):
        """Train the Naive Bayes classifier by building the vocabulary and word counts."""
        for line, sentiment in corpus:
            sentiment = int(sentiment) if sentiment in ("0", "1") else (1 if sentiment == "pos" else 0)

            tokenized_text = self.tokenizer.tokenize(line)  # Tokenize the text to get words
            self.vocab.update(tokenized_text)  # Add words to vocabulary
            self.doc_size+=1
            
            if sentiment == 1:  # Check if sentiment is positive
                self.pos_counts.update(tokenized_text)  # Update the positive word counts
                self.pos_size+=1
            else:  # Otherwise, update the negative word counts
                self.neg_counts.update(tokenized_text)
                self.neg_size+=1
    
    def compute_parameters(self) -> None:

        if self.doc_size == 0:
            raise ValueError("No training documents processed.")
        if self.pos_size == 0:
            raise ValueError("No positive documents in training set.")
        if self.neg_size == 0:
            raise ValueError("No negative documents in training set.")

        # Priors
        self.pos_logprior = math.log(self.pos_size / self.doc_size)
        self.neg_logprior = math.log(self.neg_size / self.doc_size)

        # Likelihoods
        for word in self.vocab:
            pos_count = self.pos_counts[word] + 1
            neg_count = self.neg_counts[word] + 1

            pos_denom = sum(self.pos_counts.values()) + len(self.vocab)
            neg_denom = sum(self.neg_counts.values()) + len(self.vocab)

            self.loglikelihood[(word, 1)] = math.log(pos_count / pos_denom)
            self.loglikelihood[(word, 0)] = math.log(neg_count / neg_denom)

    def predict(self, text: str) -> str:
        """Predicts class for a single document."""
        tokens = self.tokenizer.tokenize(text)
        pos_score = self.pos_logprior + sum(self.loglikelihood.get((w, 1), 0) for w in tokens)
        neg_score = self.neg_logprior + sum(self.loglikelihood.get((w, 0), 0) for w in tokens)
        return 1 if pos_score > neg_score else 0


    def evaluate(self, test_corpus: CategorizedCorpus):
        """Evaluates accuracy on a labeled corpus only using labels to compute score)."""
        correct = 0
        total = 0
        for line, true_label in test_corpus:
            pred = self.predict(line)
            if pred == int(true_label): #Normalizing - Should perhaps be done elsewhere? 
                correct += 1
            total += 1
        return correct / total if total > 0 else 0

