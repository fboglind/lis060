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
from mynltk.utils.features import get_features
from mynltk.multinomial_regression import MultinomialLogisticRegression as MLR

class MorphologicalAnalyzer:
    """A class representing a morphological analyzer using trainable models."""

    def __init__(self, training_data: list[tuple], n_iters: int = 5, model_cls=None, feature_fn=None) -> None:
        """
        Constructs a MorphologicalAnalyzer object.
        Args:
            training_data: A list of tuples containing words and their labels.
            n_iters: The number of iterations for training.
            model_cls: The classifier class to use (e.g., Perceptron or MLR).
            feature_fn: A feature extraction function (default: get_features).
        """
       
        self.n_iters = n_iters
        self.training_data = training_data
        self.model_cls = model_cls if model_cls else MLR  # store class, not instance
        self.get_features = feature_fn if feature_fn else get_features
        self.model = None  # This will later hold the actual model *instance*



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
        # Create model object
        self.model = self.model_cls(labels, self.get_features)

        # Train model object
        self.model.train(self.training_data, self.n_iters)
        # Predict words and add predictions to a list
        return [self.model.predict(word) for word in inflected_forms]

    def show_top_features(self, n: int = 10) -> None:
        """Prints top n features with the highest weights from q trained model
        Args:
            n: The number of top features to display
        """
        all_weights = []
        top_n = []

        if isinstance(self.model, MLR):
                all_weights =[(f, w) for label in self.model.labels for f, w in self.model.weights[label].items()]
                top_n = sorted(all_weights, key=lambda x: abs(x[1]), reverse=True)[:n]    
    
        if isinstance(self.model,Perceptron):
            top_n = sorted(
                self.model.theta.items(), key=lambda x: x[1], reverse=True
            )[:n]
        print("Top 10 features:")
        for idx, (key, value) in enumerate(top_n, start=1):
            print(f"{idx}. {key}: {value}")
        print()