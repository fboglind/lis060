from mynltk.utils.categorized_corpus import CategorizedCorpus
from mynltk.models.ngram_model import NGramModel
from mynltk.ngram_text_classifier import NGramTextClassifier

def main():
    """Main function to test the n-gram model on a given corpus."""
    example_sentences = tuple(
    tuple(sentence.split()) for sentence in [
        'I have never heard of a gwerkomynx .',
        'I have never heard of a horse .',
        'I never heard have of a horse .'])

    #Create test data
    imdb_train_cc = CategorizedCorpus("data/sentiment_sentences/imdb-train.tsv")
    imdb_test_cc = CategorizedCorpus("data/sentiment_sentences/imdb-test.tsv")

    with open("scores.txt", "w") as outf:
        for n in (1, 2, 3):
            print(f"Testing {n}-grams")
            model = NGramModel(
                    n, 0.001, (imdb_train_cc))


            for i, sentence in enumerate(imdb_test_cc):
                score = model.score(sentence)
                normal_score = score / len(sentence)
                print(f"{n}\t{i}\t{score:.1f}\t{normal_score:.3f}", file=outf)

            for sentence in example_sentences:
                score = model.score(sentence)
                print(f"    {score:.1f}    {' '.join(sentence)}")

    # # Create an instance of the classifier
    # imdb_classifier = NGramModel(3, 0.001, imdb_train)

    # # Train the classifier on the training corpus
    # imdb_classifier.train(imdb_train)

    # # Compute the parameters (log priors and log likelihoods)
    # imdb_classifier.compute_parameters()

    # # Evaluate the classifier on the test corpus
    # imdb_accuracy = imdb_classifier.evaluate(imdb_test)





if __name__ == '__main__':
    main()
