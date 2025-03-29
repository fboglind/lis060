"""slask.py"""

    # def read_corpus(self, corpus_dir) -> tuple[str,str]:
    #     """Read a corpus from a directory."""
    #     if corpus_dir == "sentiment_analysis": #Sentiment analysis dataset contains tsv files with either test or train in the name
    #         for corpus in os.listdir(corpus_dir):
    #             #Test or train
    #             set_type=corpus.split("-")[1].strip(".tsv")
    #             #Name of the corpus
    #             corpus-name = corpus.split("-")[0]
    #     if corpus_dir == "imdb": #IMDB dataset contains two subdirectories: neg and pos
    #         #Read the corpus directory
    #         for subdir in os.listdir(corpus_dir):
    #             #Read the subdirectory
    #             sentiment== subdir #neg or pos
    #             for filename in os.listdir(os.path.join(corpus_dir, subdir)):
    #                 category = filename.split("_")[1]
    #                 text = self.read_file(os.path.join(corpus_dir, subset, filename), category)
    #                 yield text
    #     return text, category

    #     with open(filename) as f:
    #         text = f.read()

    #     return text, category