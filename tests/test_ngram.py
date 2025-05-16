"""test_ngram.py"""

"""Test the n-gram model on a given corpus.
This script reads sentences from a gzipped file, builds an n-gram model,
and computes the log probability of sentences from a test corpus.
It also computes the normalised score of each sentence and prints
the results to a file.
"""
import gzip
from mynltk.utils.categorized_corpus import CategorizedCorpus
from mynltk.models.ngram_model import NGramModel


def sentence_reader(filename, max_sentences=None):
    """Read sentences from a gzipped file and yield them as tuples of words.
    Args:
        filename (str): The path to the gzipped file containing sentences.
        max_sentences (int, optional): The maximum number of sentences to read.
    Yields:
        tuple: A tuple representing a sentence, where each word is an element
            of the tuple.
    """
    n_sentences = 0
    with gzip.open(filename, 'rt') as f:
        for line in f:
            yield tuple(line.split())
            n_sentences += 1
            if max_sentences and n_sentences >= max_sentences:
                return


def main():
    """Main function to test the n-gram model on a given corpus."""
    # For lis050:
    # train_corpus = "/home/dsv/robe/lis050/lab1/english-train.txt.gz"
    # test_corpus = "/home/dsv/robe/lis050/lab1/english-test.txt.gz"
    # For lis060:
    # bengio_train_corpus = "data/bengio/english-train.txt.gz"
    # bengio_test_corpus = "data/bengio/english-test-1k.txt.gz"
    
    # example_sentences = tuple(
    #     tuple(sentence.split()) for sentence in [
    #         'I have never heard of a gwerkomynx .',
    #         'I have never heard of a horse .',
    #         'I never heard have of a horse .'])

    # with open("scores.txt", "w") as outf:
    #     for n in (1, 2, 3):
    #         print(f"Testing {n}-grams")
    #         model = NGramModel(
    #                 n, 0.001, sentence_reader(bengio_train_corpus))

    #         for i, sentence in enumerate(sentence_reader(bengio_test_corpus)):
    #             score = model.score(sentence)
    #             normal_score = score / len(sentence)
    #             print(f"{n}\t{i}\t{score:.1f}\t{normal_score:.3f}", file=outf)

    #         for sentence in example_sentences:
    #             score = model.score(sentence)
    #             print(f"    {score:.1f}    {' '.join(sentence)}")

    example_sentences = tuple(
    tuple(sentence.split()) for sentence in [
        'I have never heard of a gwerkomynx .',
        'I have never heard of a horse .',
        'I never heard have of a horse .'])

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
