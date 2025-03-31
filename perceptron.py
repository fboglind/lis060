"""perceptron.py - A Perceptron for morphological analysis
    Classes:
        - Perceptron: A class representing a perceptron
    Methods:
        - __init__: Constructs a Perceptron object
        - predict: Takes word and returns a predicted morphological label
        - train: Trains the Perceptron
"""
from collections import defaultdict
from collections.abc import Callable


class Perceptron:
    """A class representing a perceptron
    Attributes:
        - labels: A set of labels for the perceptron
        - theta: A dictionary mapping features to weights
        - get_features: Takes a word and a label and returns features
    """

    def __init__(self, labels: list[str], get_features: Callable) -> None:
        """Constructs a Perceptron object
        Args:
            labels: A list of labels for the perceptron
            get_features: Function that takes a word and a label and returns features
            theta: A dictionary mapping features to weights
        """
        self.get_features = get_features
        self.labels = set(labels)
        self.theta: defaultdict = defaultdict(lambda: 0)

    def predict(self, word: str) -> str:
        """Predicts the label for a given word
        Args:
            word: The word to predict the label for
            Returns:
                The predicted label for the word
        """
        # Create zero-value dictionary with labels to map .. with features
        label_dict = dict.fromkeys(self.labels, 0)

        # Iterate over all possible labels
        for label in self.labels:
            # Get all the features associated with the word
            features = self.get_features(word, label)
            # Get the sum of the values in theta of all the features
            maxed_feature_value = sum([self.theta[f] for f in features])
            # Update label in dict with summed value for features in theta
            label_dict[label] = maxed_feature_value
        # Return the label with the highest value in label_dict
        max_label = max(label_dict, key=label_dict.__getitem__)
        return max_label

    def train(self, data: list[tuple], n_iter: int = 5) -> None:
        """Trains the Perceptron
        Args:
            data: List of tuples containing words and their labels
            n_iter: The number of iterations for training
        """

        # Initialize knowledge to zero
        for word, tag in data:
            features = self.get_features(word, tag)
            for feature in features:
                self.theta[feature] = 0

        for _ in range(n_iter):
            for word, label in data:
                prediction = self.predict(word)
                features = self.get_features(word, label)
                predicted_features = self.get_features(word, prediction)
                for feature in features:
                    if feature in self.theta:
                        self.theta[feature] += 1
                    else:
                        self.theta[feature] = 1

                for feature in predicted_features:
                    if feature in self.theta:
                        self.theta[feature] -= 1
                    else:
                        self.theta[feature] = 1
