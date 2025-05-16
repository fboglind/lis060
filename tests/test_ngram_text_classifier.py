from mynltk.utils.categorized_corpus import CategorizedCorpus
from mynltk.models.ngram_model import NGramModel
from mynltk.classifier import NGramTextClassifier

def main():
    """Main function to test the n-gram model on a given corpus."""

    #Create CategorizedCorpus objects for language ID and sentiment analysis
    # imdb_train_cc = CategorizedCorpus("data/sentiment_sentences/imdb-train.tsv")
    # imdb_test_cc = CategorizedCorpus("data/sentiment_sentences/imdb-test.tsv")
    imdb_train_cc = CategorizedCorpus("data/imdb/train")
    imdb_test_cc = CategorizedCorpus("data/imdb/test")
    lang_id_train_cc = CategorizedCorpus("data/langid/langid-train.tsv")
    lang_id_test_cc = CategorizedCorpus("data/langid/langid-test.tsv")

    #Create NGramTextClassifier objects for language ID and sentiment analysis
    language_id_classifier = NGramTextClassifier(lang_id_train_cc, n=3, k=0.1, word_ngram=False)
    sentiment_classifier = NGramTextClassifier(imdb_train_cc, n=2, k=0.1, word_ngram=True)
    
    #Train the classifiers
    language_id_classifier.train(lang_id_train_cc)
    sentiment_classifier.train(imdb_train_cc)
    
    #Evaluate the classifiers
    lang_id_accuracy = language_id_classifier.evaluate(lang_id_test_cc)
    sentiment_accuracy = sentiment_classifier.evaluate(imdb_test_cc)
    print(f"Language ID Accuracy: {lang_id_accuracy:.2f}")
    print(f"Sentiment Analysis Accuracy: {sentiment_accuracy:.2f}")
    # for text, label in lang_id_test_cc:
    #     pred = language_id_classifier.predict(text)
    #     if pred != label:
    #         print(f"❌ {label=} {pred=} {text[:80]}")
    #     else:
    #         print(f"✅ {label=} {text[:80]}")


if __name__ == '__main__':
    main()
