"""morphological_analyzer.py

A simple implementation of a morphological analyzer
    Classes:
        - MorphologicalAnalyzer: A class representing
        a morphological analyzer
    Methods:
        - __init__: Constructs a MorphologicalAnalyzer object
        - predict: Makes morphological predictions for a list of words
        - get_features: Extracts features from the word form and POS
        - show_top_features: Prints features with the highest weights
          for a trained model
        - train: Trains the MorphologicalAnalyzer"""

from .models.perceptron import Perceptron


class MorphologicalAnalyzer:
    """A class representing a morpological analyser
    utilizing Perceptron objects
    """

    def __init__(self, training_data: list[tuple], n_iters: int = 5) -> None:
        """Constructs a MorphologicalAnalyzer object
        Args:
            training_data: A list of tuples containing words and their labels
            n_iters: The number of iterations for training
        """
        self.n_iters = n_iters
        self.training_data = training_data
        self.model = Perceptron

    def predict(self, test_data: list[str]) -> list[str]:
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
        print()