"""ner_analyzer.py

A simple implementation of a NER analyzer
    Classes:
        - NERAnalyzer: A class representing
        a morphological analyzer
    Methods:
        - __init__: Constructs a MorphologicalAnalyzer object
        - predict: Makes morphological predictions for a list of words
        - get_features: Extracts features from the word form and POS
        - show_top_features: Prints features with the highest weights
          for a trained model
        - train: Trains the MorphologicalAnalyzer"""
import numpy as np
from perceptron import Perceptron


class NERAnalyzer:
    """A class representing a morpological analyser
    utilizing Perceptron objects
    """

    def __init__(self, training_data: list[tuple], n_iters: int = 5) -> None:
        """Constructs a NERAnalyzer object
        Args:
            training_data: A list of tuples containing words and their labels
            n_iters: The number of iterations for training
        """
        self.n_iters = n_iters
        self.training_data = training_data
        self.model = Perceptron


     def predict(self, words: list[str]) -> list[str]:
        """Use Viterbi algorithm to find most probable sequence of tags
        for the given list of words"""
        n = len(words)
        # Viterbi table: V[t][tag] is the best log-probability up
        # to word position t ending in tag.
        V = [{} for _ in range(n + 1)]
        # Backpointers: B[t][tag] stores the previous tag that gave
        # the best probability to reach tag at position t.
        B = [{} for _ in range(n + 1)]

        # Initialize with the start symbol at position 0.
        V[0]["<s>"] = 0.0  # log probability of 1 is 0

        # Iterate over positions 1 to n
        for t in range(1, n + 1):
            word = words[t - 1].lower()
            for curr_tag in self.tags:
                best_prob = float("-inf")
                best_prev = None
                # Consider every tag at position t-1 that we reached.
                for prev_tag in V[t - 1]:
                    # Get transition probability (with a small fallback for unseen transitions)
                    trans_prob = self.tag_transitions.get(prev_tag, {}).get(
                        curr_tag, 1e-6
                    )
                    # Get emission probability (again, fallback for unknown words)
                    emiss_prob = self.tag_emissions.get(curr_tag, {}).get(
                        word, 1e-6
                    )

                    prob = (
                        V[t - 1][prev_tag]
                        + self.score(word, curr_tag, prev_tag)
                    )
                    if prob > best_prob:
                        best_prob = prob
                        best_prev = prev_tag
                if best_prev is not None:
                    V[t][curr_tag] = best_prob
                    B[t][curr_tag] = best_prev

        # Termination: transition from last tag to the end symbol '</s>'
        best_final_prob = float("-inf")
        best_final_tag = None
        for tag in V[n]:
            trans_prob = self.tag_transitions.get(tag, {}).get("</s>", 1e-6)
            prob = V[n][tag] + math.log(trans_prob)
            if prob > best_final_prob:
                best_final_prob = prob
                best_final_tag = tag

        # Backtrack to recover the best tag sequence
        best_tags = [None] * n
        current_tag = best_final_tag
        for t in range(n, 0, -1):
            best_tags[t - 1] = current_tag
            current_tag = B[t].get(current_tag, None)
        return best_tags

    def predict_morph(self, test_data: list[str]) -> list[str]:
        """Makes morphological predictions
        Args:
            test_data: A list of words to predict*
        Returns:
            A list of predicted morphological labels for the words
        """
        # Create list of inflected forms
        labels = [y for x, y in self.training_data]
        # Create a list of morphological labels
        inflected_forms = list(test_data)
        # Create perceptron object
        self.model = Perceptron(labels, self.get_features)
        # Train perceptron object
        self.model.train(self.training_data, self.n_iters)
        # Predict words and add predictions to a list
        return [self.model.predict(word) for word in inflected_forms]


    def get_ner_features(self, word: tuple, label: str) -> set[str]: #previous tag should also be in the call
        features set()
        form = form.replace(" ", "_")  # Handle multi-word expressions
        features.add(f"word.lower={word.lower()}") # lowercase form?
        features.add(f"word.is_lower={word.islower()}") # all lowercase?
        features.add(f"word.is_upper={word.isupper()}") # all caps?
        features.add(f"word.is_title={word.istitle()}") # capitalized?
        features.add(f"word.is_digit={word.isdigit()}") # is number?
        features.add(f"word.is_alpha={word.isalpha()}"}) # is all letters?
        #features.add(f"prev_tag={prev_tag}") # the previous tag
        return features

    def get_features(self, word_with_pos: tuple, label: str) -> set[str]:
        """Extract prefix/suffix features from the word form and POS
        Args:
            word_with_pos: A tuple containing the word form and its POS
            label: The morphological label
            Returns:
                A set of features for the word form and POS
        """
        form, pos = word_with_pos
        form = form.replace(" ", "_")  # Handle multi-word expressions
        features = set()

        # Generate prefixes
        for i in range(1, 6):
            if len(form) >= i:
                prefix = form[:i]
                features.add(f"{pos} prefix={prefix} tag={label}")

        # Generate suffixes
        for i in range(1, 6):
            if len(form) >= i:suffixes
        for i in range(1, 6):
            if len(form) >= i:
                suffix = form[-i:]
                features.add(f"{pos} suffix={suffix} tag={label}")

        return features

    def show_top_features(self, n: int = 10) -> None:
        """Prints top n features with the highest weights from q trained model
        Args:
            n: The number of top features to display
        """
        top_n = sorted(
            self.model.theta.items(), key=lambda x: x[1], reverse=True
        )[:n]
        print("Top 10 features:")
        for idx, (key, value) in enumerate(top_n, start=1):
            print(f"{idx}. {key}: {value}")
        p
                suffix = form[-i:]
                features.add(f"{pos} suffix={suffix} tag={label}")

        return features
