import os
import re

class CategorizedCorpus:
    """
    A class that reads a collection of texts and their categories.
    Provides a way of iterating over tuples (text, category).
    Supports both subdirectory-based IMDB format and TSV sentiment sentence format.
    
    Example:
        cc = CategorizedCorpus("data/imdb/train")
        for text, category in cc:
            print(f"{text[:40]}... {category}")
    """

    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.corpus_type = self._check_type()

    def __iter__(self):
        if self.corpus_type == "imdb":
            return self._iter_imdb()
        elif self.corpus_type == "sentiment":
            return self._iter_sentiment()
        else:
            raise ValueError("Unsupported or invalid corpus format.")

    def _check_type(self):
        if os.path.isdir(self.corpus_path):
            # Check for 'pos' and 'neg' subdirectories
            subdirs = os.listdir(self.corpus_path)
            if "pos" in subdirs and "neg" in subdirs:
                return "imdb"
        elif os.path.isfile(self.corpus_path):
            # Check if it's a valid sentiment file
            if re.match(r"^[a-zA-Z]+-(train|test)\.tsv$", os.path.basename(self.corpus_path)):
                return "sentiment"
        raise ValueError(f"{self.corpus_path} is not a valid corpus path")

    def _iter_sentiment(self):
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    text, category = line.split('\t')
                    yield text, category
                except ValueError:
                    continue  # skip malformed lines

    def _iter_imdb(self):
        for category in ["pos", "neg"]:
            category_path = os.path.join(self.corpus_path, category)
            for filename in os.listdir(category_path):
                if not filename.endswith(".txt"):
                    continue
                file_path = os.path.join(category_path, filename)
                with open(file_path, encoding='utf-8') as f:
                    text = f.read().strip()
                    yield text, category
