"""mynltk.py - My Natural Language Toolkit"""
import os
import re
import math
from typing import Iterable, Callable
from collections import defaultdict


class Tokenizer:
    """ A tokenizer that splits text into tokens using a regular expression.
    Curently the method used is the simplest possible one.
    """

    def __init__(self, pattern=None, lowercase=False):
        self.pattern = pattern or r"\w+|[^\w\s]"
        self.lowercase = lowercase
    token_specification = [
        ('NUMBER',   r'\d+(\.\d*)?'),  # Integer or decimal number
        ('ASSIGN',   r':='),           # Assignment operator
        ('END',      r';'),            # Statement terminator
        ('ID',       r'[A-Za-z]+'),    # Identifiers
        ('OP',       r'[+\-*/]'),      # Arithmetic operators
        ('NEWLINE',  r'\n'),           # Line endings
        ('SKIP',     r'[ \t]+'),       # Skip over spaces and tabs
        ('MISMATCH', r'.'),            # Any other character
    ]

    def tokenize(self, text: str, char_based=False, lowercase=False) -> list[str]:
        """For now we use a simple tokenizer"""
        if self.lowercase:
            text =  [t.lower() for t in text] # better than text.lower()
        if char_based:
            return list(text)
        return self.simple_tokenize(text)
    
    def simple_tokenize(self, text: str) -> list[str]:
        """The simplest possible implemenation of a tokenizer"""
        return re.findall(self.pattern, text)

class CategorizedCorpus:
    """A class that reads a collection of texts and their categories.
    Provides a way of iterating over tuples (text, category).
    Use the iter special method provides a convenient solution,
    that allows you to write code like this:
    
    cc = CategorizedCorpus("data/imdb/train")
    for text, category in cc:
    print(f"{text[:40]}... {category}")"""
    
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.corpus = []
        self.corpus_type=self.check_type(self.corpus_path)
        self.data = []
        
        if self.corpus_type== "sentiment":
            self._load_sentiment_sentences_format()
            
        elif self.corpus_type== "imdb":
            self._load_imdb_format()
        else:
            print("Please load a valid directory")
        
    def __iter__(self):
        return iter(self.data)
    
    def check_type(self, corpus_path):
        """Checks the type and validity of directory or file."""
        # Check if the path is a directory (i.e., IMDB format)
        if os.path.isdir(corpus_path):
            # Check if the directory is amed 'train' or 'test'
            if os.basename(corpus_path) not in ["train", "test"]:
                print(f"{corpus_path} is not a valid train/test directory")
                return None
            subdirs = os.listdir(corpus_path)
            # Check if both the neg and pos subdirectories are present
            if "pos" not in subdirs and "neg" not in subdirs:
                print(f"{corpus_path} does not contain 'pos' and 'neg' subdirectories")
                return None
               
            # Check if the subdirectories contain valid txt files
            category_path = os.path.join(subdir_path, category_sub)
            for f in os.listdir(category_path):
                if not re.match(r"^\d+_\d+\.txt$", f):
                    print(f"{category_path} contains non-format files: {f}")
                    return None
            self.corpus_type = "imdb"
            print("Format is IMDB")
            return "imdb"
            else:
                print(f"{corpus_path} does not contain 'train' and 'test' subdirectories")
                return None

        # Check if the path is a file (i.e., Sentiment format)
        elif os.path.isfile(corpus_path):
            if re.match(r"^[a-zA-Z]+-(train|test)\.tsv$", os.path.basename(corpus_path)):
                self.corpus_type = "sentiment"
                print("Format is Sentiment")
                return "sentiment"
            else:
                print(f"{corpus_path} is not a valid sentiment file")
                return None

        # If neither, it's an invalid format
        else:
            print(f"{corpus_path} is not a valid format")
            return None

    def load_sentiment_corpus(self, corpus_path) -> Iterable[tuple[str, str]]:
        """Read a sentiment corpus from a TSV file."""
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                text, category = line.strip().split('\t')
                yield text, category

    def load_imdb_corpus(self, corpus_path):
        """Read a sentiment corpus from a directory containing
        positive and negative reviews in txt files."""
        #IMDB dataset contains two subdirectories: neg and pos
        for category in os.listdir(corpus_path):
            for filename in category:
                file_path = os.path.join(corpus_path, filename)
                if os.path.isfile(file_path):
                    with open(file_path, encoding='utf-8') as f:
                        yield (f.read().strip(), int(category=="pos"))

class NGramModel:
    """A simple n-gram language model that calculates the likelihood of
    a sequence of words (a sentence) based on n-grams.

    Attributes:
        n (int): The size of the n-gram (e.g., 2 for bigrams, 3 for trigrams).
        k (float): The smoothing parameter for Laplace smoothing.
        ngram_counts (defaultdict): A dictionary that stores the
        counts of each n-gram.
        context_counts (defaultdict): A dictionary that stores the counts
        of each context
        (the first n-1 words of an n-gram).
        vocab (set): A set of all unique words in the training data.
    """

    def __init__(
        self, n: int, k: float, sentences: Iterable[tuple[str]]
    ) -> None:
        """Initialize the NGramModel with the given parameters and training
        sentences.

        Args:
            n (int): The size of the n-gram.
            k (float): The smoothing parameter for Laplace smoothing.
            sentences (Iterable[tuple[str]]): An iterable of sentences,
            where each sentence is represented as a tuple of strings.
        """
        self.n = n
        self.k = k
        self.ngram_counts: defaultdict[tuple[str, ...], int] = defaultdict(int)
        self.context_counts: defaultdict[tuple[str, ...], int] = defaultdict(
            int
        )
        self.vocab = set()

        # Process each sentence with padding
        for sentence in sentences:
            padded = ["<s>"] * (n - 1) + list(sentence) + ["</s>"]
            for word in padded:
                self.vocab.add(word)
            # Extract n-grams from padded sentence and update their counts
            for i in range(len(padded) - n + 1):
                ngram = tuple(padded[i : i + n])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

    def p(self, word: str, context: tuple[str, ...]) -> float:
        """
        Calculate the probability of a word given its context
        using Laplace smoothing.

        Args:
            word (str): The word for which the probability is to be
            calculated.
            context (tuple[str]): The context in which the word appears,
            represented as a tuple of strings.

        Returns:
            float: The probability of the word given the context.
        """
        ngram = context + (word,)
        vocab_size = len(self.vocab)
        numerator = self.ngram_counts[ngram] + self.k
        denominator = self.context_counts[context] + self.k * vocab_size
        return numerator / denominator

    def score(self, sentence: tuple[str]) -> float:
        """
        Calculate the log probability of a given sentence using n-gram model.

        This method adds padding to the sentence and computes the log
        probability of the sentence based on the n-gram probabilities.

        Args:
            sentence (tuple[str]): A tuple of words representing the
            sentence to be scored.

        Returns:
            float: The log probability of the sentence.
        """
        padded = ["<s>"] * (self.n - 1) + list(sentence) + ["</s>"]
        log_prob = 0.0
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - self.n + 1 : i])
            prob = self.p(padded[i], context)
            log_prob += math.log(prob) if prob > 0 else -float("inf")
        return log_prob

class BPEncoder:
    pass

class NFoldCrossValidator:
    pass

class Perceptron:
    """A class representing a perceptron"""
    def __init__(self, labels: list[str], get_features: Callable) -> None:
        """Constructs a Perceptron object"""
        self.get_features = get_features
        self.labels = set(labels)
        self.theta: defaultdict = defaultdict(lambda: 0)

    def predict(self, word: str) -> str:
        """Takes a word and returns a predicted morphological label"""
        # Create zero-value dictionary with labels to map .. with features
        label_dict = dict.fromkeys(self.labels, 0)

        # Iterate over all possible labels
        for label in self.labels:
            # Get all the features associated with the word
            features = self.get_features(word, label)
            # Get the sum of the values in theta of all the features
            maxed_feature_value = sum([self.theta[f] for f in features])
            # Update label in dict with summed value for features in theta
            label_dict[label] = maxed_feature_value
        # Return the label with the highest value in label_dict
        max_label = max(label_dict, key=label_dict.__getitem__)
        return max_label

    def train(self, data: list[tuple], n_iter: int = 5) -> None:
        """Trains the Perceptron"""

        # Initialize knowledge to zero
        for word, tag in data:
            features = self.get_features(word, tag)
            for feature in features:
                self.theta[feature] = 0

        for _ in range(n_iter):
            for word, label in data:
                prediction = self.predict(word)
                features = self.get_features(word, label)
                predicted_features = self.get_features(word, prediction)
                for feature in features:
                    if feature in self.theta:
                        self.theta[feature] += 1
                    else:
                        self.theta[feature] = 1

                for feature in predicted_features:
                    if feature in self.theta:
                        self.theta[feature] -= 1
                    else:
                        self.theta[feature] = 1

class StructuredPerceptron:
    pass

class Analyzer:
    """A class representing a morpological analyser utilizing
    Perceptron objects
    """

    def __init__(self, training_data: list[tuple], n_iters: int = 5) -> None:
        """Constructs an Analyzer object"""
        self.n_iters = n_iters
        self.training_data = training_data

    def predict(self, test_data: list[str]) -> list[str]:
        """Takes a list of words and returns a list of
        morphological predictions
        """
        # Create list of inflected forms
        labels = [y for x, y in self.training_data]
        # Create a list of morphological labels
        inflected_forms = list(test_data)
        # Create perceptron object
        perc = Perceptron(labels, self.get_features)
        # Train perceptron object
        perc.train(self.training_data, self.n_iters)
        # Predict words and add predictions to a list
        predicted_labels = [perc.predict(word) for word in inflected_forms]
        # Show top ten
        top_10 = sorted(self.theta.items(), key=lambda x: x[1], reverse=True)[:10]
        print("Top 10 features:")
        for idx, (key, value) in enumerate(top_10, start=1):
            print(f"{idx}. {key}: {value}")
        print()
        # Return predicted list of predicted morphological labels
        return predicted_labels

    def get_features(self, word: str, label: str) -> list[str]:
        """Takes word and a label, returns a list of strings of suffixes
        of lengths 1 to 5 concatenated with labels
        """
        features = [word[- i:] + "&" + label for i in range(1, 6)]
        return features

class FeatureBasedMorphologicalAnalyzer:
    pass

class NER:
    """Named Entity Recognition with Perceptron"""
    pass

class SentimentAnalysis:
    pass


def main():
    pass
if __name__ == "__main__":
    main()