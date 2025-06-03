from collections import defaultdict
import math
from random import shuffle
from typing import Callable
#from mynltk.utils.features import get_features


class MultinomialLogisticRegression:
    def __init__(self, labels: list[str], get_features: Callable, learning_rate=0.1):
        self.labels = set(labels)
        self.get_features = get_features
        self.learning_rate = learning_rate
        self.weights = defaultdict(lambda: defaultdict(float))

    def train(self, training_data: list[tuple[tuple[str, str], str]], n_iters: int = 5):
        for epoch in range(n_iters):
            shuffle(training_data)
            for i, ((word, pos), gold_label) in enumerate(training_data):
                features = self.get_features((word, pos), gold_label)

                probs = self.predict_proba(features)
                for label in self.labels:
                    error = (1.0 if label == gold_label else 0.0) - probs[label]
                    if abs(error) > 1e-6:  # Only update if there's meaningful error
                        for f in features:                        
                            self.weights[label][f] += self.learning_rate * error
                        #debuging
                        # old_value = self.weights[label][f]
                        # if error != 0.0 and epoch == 0 and i < 5:
                        #     print(f"[Epoch {epoch}] Updated weight for label={label}, feature={f}: {old_value:.4f} → {self.weights[label][f]:.4f}")
                #debugging
                if epoch == 0 and i < 5:
                    print(f"[Epoch {epoch}] Sample features for ({word}, {pos}): {features}")



    def predict_proba(self, features: set[str]) -> dict[str, float]:
        scores = {
            label: sum(self.weights[label][f] for f in features)
            for label in self.labels
        }
        max_score = max(scores.values(), default=0)
        exp_scores = {l: math.exp(v - max_score) for l, v in scores.items()}
        total = sum(exp_scores.values()) or 1.0
        return {l: v / total for l, v in exp_scores.items()}

    # def predict(self, word_with_pos: tuple[str, str]) -> str:
    #     features = self.get_features(word_with_pos, "")
    #     probs = self.predict_proba(features)
    #     return max(probs.items(), key=lambda x: x[1])[0]

    def predict(self, word_with_pos: tuple[str, str]) -> str:
        """Predict by trying all possible labels and choosing the best one"""
        best_label = None
        best_prob = -1
        
        for label in self.labels:
            # Generate features WITH the label (same as training)
            features = self.get_features(word_with_pos, label)
            probs = self.predict_proba(features)
            
            if probs[label] > best_prob:
                best_prob = probs[label]
                best_label = label
        
        return best_label


    # def top_features(self, label: str, k: int = 10):
    #     return sorted(self.weights[label].items(), key=lambda x: abs(x[1]), reverse=True)[:k]
    

    def show_top_features(self, n: int = 10) -> None:
        """Prints top n features with the highest weights from q trained model
        Args:
            n: The number of top features to display
        """

        all_weights = [(f, w) for label in self.labels for f, w in self.weights[label].items()]
        top_n = sorted(all_weights, key=lambda x: abs(x[1]), reverse=True)[:n]

        print("Top 10 features:")
        for idx, (key, value) in enumerate(top_n, start=1):
            print(f"{idx}. {key}: {value}")
        print()