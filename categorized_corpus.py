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