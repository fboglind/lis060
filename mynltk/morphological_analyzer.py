"""morphological_analyzer.py

A simple implementation of a morphological analyzer
    Classes:
        - MorphologicalAnalyzer: A class representing
        a morphological analyzer
    Methods:
        - __init__: Constructs a MorphologicalAnalyzer object
        - predict: Makes morphological predictions for a list of words
        - show_top_features: Prints features with the highest weights
          for a trained model
"""

from .models.perceptron import Perceptron
from mynltk.utils.features import get_features_mlr, get_features_perceptron
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
            feature_fn: A feature extraction function (default: get_features)."""
        self.n_iters = n_iters
        self.training_data = training_data
        
        if model_cls is None:
            self.model_cls = MLR
        else:
            self.model_cls = model_cls
            
        # Use appropriate feature function based on model
        if feature_fn is None:
            if self.model_cls == MLR:
                self.get_features = get_features_mlr
            else:
                self.get_features = get_features_perceptron
        else:
            self.get_features = feature_fn
            
        self.model = None

    def predict(self, test_data: list[str]) -> list[str]:
        """Makes morphological predictions
        Args:
            test_data: A list of words to predict
        Returns:
            A list of predicted morphological labels for the words
        """
        # Extract labels from training data
        labels = [y for x, y in self.training_data]
        
        # Debug: Check what we're working with
        print(f"Creating model of type: {self.model_cls}")
        print(f"Number of unique labels: {len(set(labels))}")
        print(f"Feature function: {self.get_features}")
        
        # Create model instance
        self.model = self.model_cls(labels, self.get_features)
        
        # Train the model
        print(f"Training for {self.n_iters} iterations...")
        self.model.train(self.training_data, self.n_iters)
        
        # Make predictions
        print("Making predictions...")
        return [self.model.predict(word) for word in test_data]

    def show_top_features(self, n: int = 10) -> None:
        """Prints top n features with the highest weights from a trained model
        Args:
            n: The number of top features to display
        """
        if self.model is None:
            print("No trained model available. Train first by calling predict().")
            return
            
        all_weights = []
        top_n = []

        if isinstance(self.model, MLR):
            all_weights = [(f, w) for label in self.model.labels 
                          for f, w in self.model.weights[label].items()]
            top_n = sorted(all_weights, key=lambda x: abs(x[1]), reverse=True)[:n]    
    
        elif isinstance(self.model, Perceptron):
            top_n = sorted(
                self.model.theta.items(), key=lambda x: x[1], reverse=True
            )[:n]
            
        print(f"Top {n} features:")
        for idx, (key, value) in enumerate(top_n, start=1):
            print(f"{idx}. {key}: {value:.4f}")
        print()