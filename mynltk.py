"""My Natural Language Toolkit"""
import os
import re

class Tokenizer:
    """
    """

    def __init__(self, pattern=None):
        self.pattern = pattern or r"\w+|[^\w\s]"

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

    def tokenize(self, text: str) -> list[str]:
        """For now we use a simple tokenizer"""
        return self.simple_tokenize(text)
    
    def simple_tokenize(self, text: str) -> list[str]:
        """The simplest possible implemenation of a tokenizer"""
        return re.findall(self.pattern, text)

class CategorizedCorpus:
    """Your class should provide a way of iterating over tuples (text, category).
    Use the iter special method provides a convenient solution,
    that allows you to write code like this:
    cc = CategorizedCorpus("data/imdb/train")
    for text, category in cc:
    print(f"{text[:40]}... {category}")"""
    
    def __init__(self):
        pass


    def read_corpus(self, corpus_dir) -> tuple[str,str]:
        """Read a corpus from a directory."""
        if corpus_dir == "sentiment_analysis": #Sentiment analysis dataset contains tsv files with either test or train in the name
            for corpus in os.listdir(corpus_dir):
                #Test or train
                set_type=corpus.split("-")[1].strip(".tsv")
                #Name of the corpus
                corpus-name = corpus.split("-")[0]
        if corpus_dir == "imdb": #IMDB dataset contains two subdirectories: neg and pos
            #Read the corpus directory
            for subdir in os.listdir(corpus_dir):
                #Read the subdirectory
                sentiment== subdir #neg or pos
                for filename in os.listdir(os.path.join(corpus_dir, subdir)):
                    category = filename.split("_")[1]
                    text = self.read_file(os.path.join(corpus_dir, subset, filename), category)
                    yield text


        with open(filename) as f:
            text = f.read()

        return text, category


class BPEncoder:
    pass

class NFoldCrossValidator:
    pass

class Perceptron:
    pass

class StructuredPerceptron:
    pass

class FeatureBasedMorphologicalAnalyzer:
    pass


class NER:
    """Named Entity Recognition with Perceptron"""
    pass

class SentimentAnalysis:
    pass


