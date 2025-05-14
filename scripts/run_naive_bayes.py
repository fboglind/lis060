"""run_naive_bayes.py"""

from mynltk.utils.categorized_corpus import CategorizedCorpus
from mynltk.naive_bayes_text_classifier import NaiveBayesTextClassifier

def main() -> None:
    # Load the training and test corpora
    #imdb_train = CategorizedCorpus("data/imdb/train")
    #imdb_test = CategorizedCorpus("data/imdb/test")
    imdb_train = CategorizedCorpus("data/sentiment_sentences/imdb-train.tsv")
    imdb_test = CategorizedCorpus("data/sentiment_sentences/imdb-test.tsv")
    amazon_train = CategorizedCorpus("data/sentiment_sentences/amazon-train.tsv")
    amazon_test = CategorizedCorpus("data/sentiment_sentences/amazon-test.tsv")
    yelp_train = CategorizedCorpus("data/sentiment_sentences/yelp-train.tsv")
    yelp_test = CategorizedCorpus("data/sentiment_sentences/yelp-test.tsv")

    # Create an instance of the classifier
    imdb_classifier = NaiveBayesTextClassifier()
    amazon_classifier = NaiveBayesTextClassifier()
    yelp_classifier = NaiveBayesTextClassifier()

    # Train the classifier on the training corpus
    imdb_classifier.train(imdb_train)
    #print(f"[DEBUG] doc_size={imdb_classifier.doc_size}, pos_size={imdb_classifier.pos_size}, neg_size={imdb_classifier.neg_size}")
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
