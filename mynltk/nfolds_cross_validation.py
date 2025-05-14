"""cross_validation.py"""
#from categorized_corpus import CategorizedCorpus
from typing import TypeVar, Generic, Callable
#from naive_bayes_text_classifier import NaiveBayesTextClassifier
from mynltk.naive_bayes_text_classifier import NaiveBayesTextClassifier


T = TypeVar('T')

class CrossValidation(Generic[T]):
    """A class for cross-validation"""
    def __init__(self):
        self.folds =[]  # List to hold the folds
        self.test_set = None  # Placeholder for test corpus
        self.training_set = None  # Placeholder for training corpus

    def n_fold(self, data_points: list[T], evaluation_function: Callable[[list[T], list[T]], float], n: int) -> list[float]:
        #A generic function (...) where where data points is a list or tuple of data points, 
        # n is the number of folds.
        n = n or self.n #!?
        fold_size = len(data_points) // n
        scores = []
        
        for i in range(n):
            # Split data into training and test sets
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i != n - 1 else len(data_points)
            self.test_set = data_points[test_start:test_end]
            self.training_set = data_points[:test_start] + data_points[test_end:]
            
            # Evaluate using the provided function
            score = evaluation_function(self.training_set, self.test_set)
            scores.append(score)
            
        return scores

    @staticmethod
    def evaluate_fold(training_set: list[tuple[str, int]], test_set: list[tuple[str, int]]) -> float:
        """Evaluation function for cross-validation folds"""
        classifier = NaiveBayesTextClassifier()
        classifier.train(training_set)
        classifier.compute_parameters()
        
        correct = 0
        for text, true_label in test_set:
            if classifier.predict(text) == int(true_label):
                correct += 1
        return correct / len(test_set) if test_set else 0
