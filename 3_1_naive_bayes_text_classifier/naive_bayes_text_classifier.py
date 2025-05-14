"""naive_bayes_text_classifier.py
A bag-of-words Naive Bayes text classifier based on equations (4.10)
and (4.14) in Jurafsky and Martin (2024, Chapter 4). Test using tokenizer and
CategorizedCorpus classes.
"""
from collections import Counter
import math
from categorized_corpus import CategorizedCorpus
from tokenizer import Tokenizer


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
        
        
    def get_logliklihood(self):
        for word in self.vocab:
            # Calculate likelihood for positive class
            pos_word_count = self.pos_counts[word] + 1  # Add-one smoothing
            pos_likelihood = pos_word_count / (self.pos_size + len(self.vocab))
            self.loglikelihood[(word, 1)] = math.log(pos_likelihood)

            # Calculate likelihood for negative class
            neg_word_count = self.neg_counts[word] + 1  # Add-one smoothing
            neg_likelihood = neg_word_count / (self.neg_size + len(self.vocab))
            self.loglikelihood[(word, 0)] = math.log(neg_likelihood)
        
    def train(self, corpus: CategorizedCorpus):
        """Train the Naive Bayes classifier by building the vocabulary and word counts."""
        for line, sentiment in corpus:
            tokenized_text = self.tokenizer.tokenize(line)  # Tokenize the text to get words
            self.vocab.update(tokenized_text)  # Add words to vocabulary
            self.doc_size+=1

            if sentiment == 1:  # Check if sentiment is positive
                self.pos_counts.update(tokenized_text)  # Update the positive word counts
                self.pos_size+=1
            else:  # Otherwise, update the negative word counts
                self.neg_counts.update(tokenized_text)
                self.neg_size+=1
        #print(f"[train] Docs: total={self.doc_size}, pos={self.pos_size}, neg={self.neg_size}")
    
    def compute_parameters(self) -> None:
        if self.doc_size == 0:
            raise ValueError("No data provided for training.")

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
        
    def predict_corpus(self, corpus: CategorizedCorpus) -> list[tuple[str, str]]:
        """Predicts sentiment for each sentence in the corpus."""
        predictions = []
        for line, _ in corpus:
            sentiment = self.predict(line)
            predictions.append((line, sentiment))
        return predictions

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
            if pred == true_label:
                correct += 1
            total += 1
        return correct / total if total > 0 else 0
    
    @staticmethod
    def evaluation_function(test_set: list[tuple[str, str]], test_set_labels: list[str]) -> float:
        """Evaluates accuracy on a labeled corpus (but only uses labels to compute score)."""
        correct = 0
        total = 0
        for line, true_label in zip(test_set, test_set_labels):
            pred = NaiveBayesTextClassifier.predict(line)
            if pred == true_label:
                correct += 1
            total += 1
        return correct / total if total > 0 else 0
    
    @staticmethod
    def evaluate_naive_bayes(training_set: list[tuple[str, int]], 
                            test_set: list[tuple[str, int]]) -> float:
        classifier = NaiveBayesTextClassifier()
        classifier.train(training_set)
        classifier.compute_parameters()

        correct = 0
        total = 0
        for text, true_label in test_set:
            prediction = classifier.predict(text)
            if prediction == true_label:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0
    
    @staticmethod
    def evaluate_fold(training_set: list[tuple[str, int]], test_set: list[tuple[str, int]]) -> float:
        """Static evaluation function for cross-validation folds"""
        classifier = NaiveBayesTextClassifier()
        classifier.train(training_set)  # Works with raw lists
        classifier.compute_parameters()
        
        correct = 0
        for text, true_label in test_set:
            if classifier.predict(text) == true_label:
                correct += 1
        return correct / len(test_set) if test_set else 0

def main() -> None:
    """Main function"""
    # Load the training and test corpora
    #imdb_train = CategorizedCorpus("../data/imdb/train")
    #imdb_test = CategorizedCorpus("../data/imdb/test")
    imdb_train = CategorizedCorpus("../data/sentiment_sentences/imdb-train.tsv")
    imdb_test = CategorizedCorpus("../data/sentiment_sentences/imdb-test.tsv")
    amazon_train = CategorizedCorpus("../data/sentiment_sentences/amazon-train.tsv")
    amazon_test = CategorizedCorpus("../data/sentiment_sentences/amazon-test.tsv")
    yelp_train = CategorizedCorpus("../data/sentiment_sentences/yelp-train.tsv")
    yelp_test = CategorizedCorpus("../data/sentiment_sentences/yelp-test.tsv")
    
    # Create an instance of the classifier
    imdb_classifier = NaiveBayesTextClassifier()
    amazon_classifier = NaiveBayesTextClassifier()
    yelp_classifier = NaiveBayesTextClassifier()
    
    # Train the classifier on the training corpus
    imdb_classifier.train(imdb_train)
    amazon_classifier.train(amazon_train)
    yelp_classifier.train(yelp_train)
    # Compute the parameters (log priors and log likelihoods)
    imdb_classifier.compute_parameters()
    amazon_classifier.compute_parameters()
    yelp_classifier.compute_parameters()
    
    # Evaluate the classifier on the test corpus
    imdb_accuracy = imdb_classifier.evaluate(imdb_test)
    amazon_accuracy = amazon_classifier.evaluate(amazon_test)
    yelp_accuracy = yelp_classifier.evaluate(yelp_test)
    
    print(f"Accuracy (imdb) : {imdb_accuracy:.2f}")
    print(f"Accuracy (amazon) : {amazon_accuracy:.2f}")
    print(f"Accuracy (yelp) : {yelp_accuracy:.2f}")
if __name__=="__main__":
    main()