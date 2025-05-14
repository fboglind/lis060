"""3_4_ngram_text_classifier.py
Perform text classification using one n-gram language model for each category.
Make the class general to handle these types of problems:
1. Language classification using character a 3-gram model
2. Sentiment analysis using a word n-gram model"""
from mynltk.models.ngram_model import NGramModel
from mynltk import CategorizedCorpus

class NGramTextClassifier:
    def __init__(self, corpus):
        pass


def main():
    cc_imdb = CategorizedCorpus("data/imdb/train")
    cc_sentiment = CategorizedCorpus("../data/sentiment/imbd-train.tsv") #?
    for text, category in cc_imdb:
        print(f"{text[:40]}... {category}")
if __name__=="__main__":
    main()