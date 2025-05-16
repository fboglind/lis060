"""cross_validation.py"""
from mynltk import CategorizedCorpus
from typing import TypeVar, Generic, Callable
from mynltk.naive_bayes_text_classifier import NaiveBayesTextClassifier

T = TypeVar('T')

class CrossValidation(Generic[T]):
    """A class for cross-validation"""
    def __init__(self) -> None:
        """Initialize the cross-validation class"""
        self.folds =[]
        self.test_set = None 
        self.training_set = None
        
    def n_fold(self, data_points: list[T], evaluation_function: Callable[[list[T], list[T]], float], n: int) -> list[float]:
        #A generic function (...) where where data points is a list or tuple of data points, n is the number of folds.
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
            if classifier.predict(text) == int(true_label): #ad hoc normalization..
                correct += 1
        return correct / len(test_set) if test_set else 0

def main() -> None:
    """The main function"""
    # Example usage
    #corpus_path = "../data/imdb/test" #[0.8072, 0.8, 0.786, 0.8004, 0.7956, 0.8784, 0.8696, 0.8716, 0.8748, 0.8796], Mean: 0.8363
    #corpus_path = "../data/imdb/train" # [0.7816, 0.7856, 0.7876, 0.7852, 0.7928, 0.8656, 0.862, 0.8664, 0.8672, 0.862], Mean:  0.8256
    #corpus_path = "../data/sentiment_sentences/imdb-test.tsv" #[0.7, 0.74, 0.76, 0.82, 0.74, 0.8, 0.8, 0.68, 0.78, 0.68], Mean: 0.7500
    corpus_path = "../data/sentiment_sentences/imdb-train.tsv" # [0.7, 0.72, 0.8, 0.7, 0.7, 0.74, 0.82, 0.72, 0.76, 0.66], Mean: 0.7320
    cc = CategorizedCorpus(corpus_path)
    data_points = list(cc)  # Convert the generator to a list
    cross_val = CrossValidation()
    cross_val.n_fold(data_points, CrossValidation.evaluate_fold, n=10) # pass static method as arg
    scores = cross_val.n_fold(
        data_points, 
        CrossValidation.evaluate_fold,
        n=10
    )
    
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {sum(scores)/len(scores):.4f}")
    
if __name__ == "__main__":
    main()
