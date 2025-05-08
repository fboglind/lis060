"""naive_bayes_text_classifier.py
A bag-of-words Naive Bayes text classifier based on equations (4.10)
and (4.14) in Jurafsky and Martin (2024, Chapter 4). Test using tokenizer and
CategorizedCorpus classes.
"""
from categorized_corpus import CategorizedCorpus
import numpy as np
from tokenizer import Tokenizer

class NaiveBayesTextClassifier:
    """A classifier"""
    def __init__(self, corpus: CategorizedCorpus) -> None:
        self.corpus=corpus
        self.bag_of_words={Tokenizer.tokenize(word) for word in self.corpus}









def main() -> None:
    """Main function"""
    imdb_train = CategorizedCorpus("data/imdb/train")

if __name__=="__main__":
    main()