"""classifier.py"""
"""An abstract base class for text classifiers"""
from abc import ABC, abstractmethod
from collections import Counter
import math

from mynltk.models.ngram_model import NGramModel
from mynltk import CategorizedCorpus
from mynltk import Tokenizer

class Classifier(ABC):

    @abstractmethod
    def train(self, training_set: CategorizedCorpus) -> None:
        """Fit the classifier to the training data."""
        pass

    @abstractmethod
    def compute_parameters(self) -> None:
        """Compute probabilities or weights from training statistics."""
        pass

    @abstractmethod
    def predict(self, text: str) -> int:
        """Predict the class label for a single instance."""
        pass

    def evaluate(self, test_set: CategorizedCorpus) -> float:
        """Evaluate accuracy on a labeled test set."""
        correct = 0
        for text, label in test_set:
            if self.predict(text) == label:
                correct += 1
        return correct / len(test_set) if test_set else 0

class NGramTextClassifier(Classifier):
    """3-4 n-gram text classifier.
    Perform text classification using one n-gram language model for each category.
    Make the class general to handle these types of problems:
    1. Language classification using character a 3-gram model
    2. Sentiment analysis using a word n-gram model
    """
    def __init__(self, training_set: CategorizedCorpus,n: int = 1, k: float = 0.001, word_ngram=False ) -> None:
        """Initialize the classifier with an n-gram model for each category."""
        self.n = n
        self.k = k
        self.word_ngram = word_ngram
        self.ngram_dict = {}
        self.category_sentences = {}
        self.tokenizer = Tokenizer()


    def train(self, training_set: CategorizedCorpus) -> None:
        """Fit the classifier to the training data."""
        for text, category in training_set:
            
            if category not in self.category_sentences:
                self.category_sentences[category] = []
            if self.word_ngram:
                # Use words as tokens
                tokens = self.tokenizer.tokenize(text)
            else: 
                
            self.category_sentences[category].append(tuple(tokens))

        for category, sentences in self.category_sentences.items():
            self.ngram_dict[category] = NGramModel(self.n, self.k, sentences)

    def compute_parameters(self) -> None:
        """Compute probabilities or weights from training statistics."""
        pass

    def predict(self, text: str) -> str:
        tokens = self.tokenizer.tokenize(text) 
        sentence = tuple(tokens)
        best_category = max(
            self.ngram_dict.keys(),
            key=lambda cat: self.ngram_dict[cat].score(sentence)
        )
        return best_category

class NaiveBayesTextClassifier(Classifier):
    """A classifier that uses the Naive Bayes algorithm for text classification."""
    def __init__(self) -> None:
        """Initialize the classifier with a corpus"""        
        self.tokenizer = Tokenizer()
        self.vocab=set()
        self.pos_counts = Counter()
        self.neg_counts = Counter()
        self.doc_size=0
        self.pos_size=0
        self.neg_size=0
        self.pos_logprior = None
        self.neg_logprior = None
        self.loglikelihood = {}


    def train(self, corpus: CategorizedCorpus) -> None:
        """Train the Naive Bayes classifier by building the vocabulary and word counts.
        Args:
            corpus (CategorizedCorpus): A categorized corpus containing text and labels.
        Returns:
            None
        """
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
        """Compute the parameters of the Naive Bayes classifier.
        This includes calculating the log prior probabilities and log likelihoods for each word.
        Raises:
            ValueError: If no training documents have been processed or if there are no positive or negative documents.
        Returns:
            None
        """

        if self.doc_size == 0:
            raise ValueError("No training documents processed.")
        if self.pos_size == 0:
            raise ValueError("No positive documents in training set.")
        if self.neg_size == 0:
            raise ValueError("No negative documents in training set.")

        # Priors
        self.pos_logprior = math.log(self.pos_size / self.doc_size) # Probability of positive class
        self.neg_logprior = math.log(self.neg_size / self.doc_size) # Probability of negative class

        # Likelihoods
        for word in self.vocab:
            pos_count = self.pos_counts[word] + 1 #Likelihood of word given positive class (+1 for smoothing)
            neg_count = self.neg_counts[word] + 1 #Likelihood of word given negative class

            pos_denom = sum(self.pos_counts.values()) + len(self.vocab)
            neg_denom = sum(self.neg_counts.values()) + len(self.vocab)

            self.loglikelihood[(word, 1)] = math.log(pos_count / pos_denom) # Likelihood of word given positive class
            self.loglikelihood[(word, 0)] = math.log(neg_count / neg_denom)

    def predict(self, text: str) -> str:
        """Predicts class for a single document.
        Args:
            text (str): The text to classify.
        Returns:
            str: The predicted class label (0 or 1).
        """
        tokens = self.tokenizer.tokenize(text)
        pos_score = self.pos_logprior + sum(self.loglikelihood.get((w, 1), 0) for w in tokens)
        neg_score = self.neg_logprior + sum(self.loglikelihood.get((w, 0), 0) for w in tokens)
        return 1 if pos_score > neg_score else 0


    def evaluate(self, test_corpus: CategorizedCorpus) -> float:
        """Evaluates accuracy on a labeled corpus only using labels to compute score).
        Args:
            test_corpus (CategorizedCorpus): A categorized corpus containing text and labels.
        Returns:
            float: The accuracy of the classifier on the test corpus.
        """
        correct = 0
        total = 0
        for line, true_label in test_corpus:
            pred = self.predict(line)
            if pred == int(true_label): #Normalizing - Should perhaps be done elsewhere? 
                correct += 1
            total += 1
        return correct / total if total > 0 else 0
