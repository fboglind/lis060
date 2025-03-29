"""ngram model.py contains the NGramModel class"""
from typing import Iterable
from collections import defaultdict
import math


class NGramModel:
    """A simple n-gram language model that calculates the likelihood of
    a sequence of words (a sentence) based on n-grams.

    Attributes:
        n (int): The size of the n-gram (e.g., 2 for bigrams, 3 for trigrams).
        k (float): The smoothing parameter for Laplace smoothing.
        ngram_counts (defaultdict): A dictionary that stores the
        counts of each n-gram.
        context_counts (defaultdict): A dictionary that stores the counts
        of each context
        (the first n-1 words of an n-gram).
        vocab (set): A set of all unique words in the training data.
    """

    def __init__(
        self, n: int, k: float, sentences: Iterable[tuple[str]]
    ) -> None:
        """Initialize the NGramModel with the given parameters and training
        sentences.

        Args:
            n (int): The size of the n-gram.
            k (float): The smoothing parameter for Laplace smoothing.
            sentences (Iterable[tuple[str]]): An iterable of sentences,
            where each sentence is represented as a tuple of strings.
        """
        self.n = n
        self.k = k
        # self.ngram_counts: defaultdict[tuple[str, int]] = defaultdict(int)
        # self.context_counts: defaultdict[tuple[str, int]] = defaultdict(int)
        self.ngram_counts: defaultdict[tuple[str, ...], int] = defaultdict(int)
        self.context_counts: defaultdict[tuple[str, ...], int] = defaultdict(
            int
        )
        self.vocab = set()

        # Process each sentence with padding
        for sentence in sentences:
            padded = ["<s>"] * (n - 1) + list(sentence) + ["</s>"]
            for word in padded:
                self.vocab.add(word)
            # Extract n-grams from padded sentence and update their counts
            for i in range(len(padded) - n + 1):
                ngram = tuple(padded[i : i + n])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

    def p(self, word: str, context: tuple[str, ...]) -> float:
        """
        Calculate the probability of a word given its context
        using Laplace smoothing.

        Args:
            word (str): The word for which the probability is to be
            calculated.
            context (tuple[str]): The context in which the word appears,
            represented as a tuple of strings.

        Returns:
            float: The probability of the word given the context.
        """
        ngram = context + (word,)
        vocab_size = len(self.vocab)
        numerator = self.ngram_counts[ngram] + self.k
        denominator = self.context_counts[context] + self.k * vocab_size
        return numerator / denominator

    def score(self, sentence: tuple[str]) -> float:
        """
        Calculate the log probability of a given sentence using n-gram model.

        This method adds padding to the sentence and computes the log
        probability of the sentence based on the n-gram probabilities.

        Args:
            sentence (tuple[str]): A tuple of words representing the
            sentence to be scored.

        Returns:
            float: The log probability of the sentence.
        """
        padded = ["<s>"] * (self.n - 1) + list(sentence) + ["</s>"]
        log_prob = 0.0
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - self.n + 1 : i])
            prob = self.p(padded[i], context)
            log_prob += math.log(prob) if prob > 0 else -float("inf")
        return log_prob
