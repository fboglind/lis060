from collections import defaultdict
import math
from random import shuffle
from typing import Callable


class MultinomialLogisticRegression:
    def __init__(self, labels: list[str], get_features: Callable, learning_rate=0.1):
        self.labels = set(labels)
        self.get_features = get_features
        self.learning_rate = learning_rate
        self.weights = defaultdict(lambda: defaultdict(float))

    def train(self, training_data: list[tuple[tuple[str, str], str]], n_iters: int = 5):
        batch_size = 1000  # Process in batches to manage memory
        
        for epoch in range(n_iters):
            shuffle(training_data)
            
            for batch_start in range(0, len(training_data), batch_size):
                batch_end = min(batch_start + batch_size, len(training_data))
                batch = training_data[batch_start:batch_end]
                
                # Process batch
                for (word, pos), gold_label in batch:
                    features = self.get_features((word, pos), None)
                    probs = self.predict_proba(features)
                    
                    for label in self.labels:
                        error = (1.0 if label == gold_label else 0.0) - probs[label]
                        if abs(error) > 1e-6:
                            for f in features:
                                self.weights[label][f] += self.learning_rate * error
                
                # Optional: print progress
                if batch_start % 10000 == 0:
                    print(f"Epoch {epoch+1}/{n_iters}, processed {batch_end}/{len(training_data)} examples")

    def predict(self, word_with_pos: tuple[str, str]) -> str:
        # Use label-agnostic features
        features = self.get_features(word_with_pos, None)  # Pass None for label
        probs = self.predict_proba(features)
        return max(probs.items(), key=lambda x: x[1])[0]

    def predict_proba(self, features: set[str]) -> dict[str, float]:
        scores = {
            label: sum(self.weights[label][f] for f in features)
            for label in self.labels
        }
        max_score = max(scores.values(), default=0)
        exp_scores = {l: math.exp(v - max_score) for l, v in scores.items()}
        total = sum(exp_scores.values()) or 1.0
        return {l: v / total for l, v in exp_scores.items()}

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