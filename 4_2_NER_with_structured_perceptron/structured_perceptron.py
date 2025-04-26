"""structured_perceptron.py

A structured perceptron implementation for Named Entity Recognition (NER)
    Classes:
        - StructuredPerceptron: A class representing a structured perceptron
        - NERTagger: A class for NER tagging using the structured perceptron
"""
from collections import defaultdict
from collections.abc import Callable


class StructuredPerceptron:
    """A class representing a structured perceptron for sequence labeling
    
    Attributes:
        - labels: A set of labels for the perceptron
        - theta: A dictionary mapping features to weights
        - get_features: Takes a word, label, and context and returns features
    """

    def __init__(self, labels: set, get_features: Callable) -> None:
        """Constructs a StructuredPerceptron object
        
        Args:
            labels: A set of labels for the perceptron
            get_features: Function to extract features
        """
        self.get_features = get_features
        self.labels = labels
        self.theta = defaultdict(float)  # Default weight is 0
        
    def score(self, features: set) -> float:
        """Compute score for a set of features
        
        Args:
            features: Set of feature strings
            
        Returns:
            The sum of weights for the features
        """
        return sum(self.theta[feature] for feature in features)
    
    def predict_next_tag(self, word: str, prev_tag: str, sentence=None, position=None) -> str:
        """Predict the next tag in the sequence using greedy search
        
        Args:
            word: The current word
            prev_tag: The previous tag
            sentence: The full sentence
            position: Position in the sentence
            
        Returns:
            The predicted tag with highest score
        """
        best_score = float('-inf')
        best_tag = None
        
        # Try each possible tag and select the one with highest score
        for tag in self.labels:
            features = self.get_features(word, tag, prev_tag, sentence, position)
            score = self.score(features)
            
            if score > best_score:
                best_score = score
                best_tag = tag
                
        return best_tag
        
    def predict_sequence(self, words: list) -> list:
        """Predict tags for a sequence of words using greedy search
        
        Args:
            words: List of words in the sequence
            
        Returns:
            List of predicted tags
        """
        tags = []
        prev_tag = "O"  # Start with outside tag
        
        for i, word in enumerate(words):
            # Predict next tag based on the word and previous tag
            next_tag = self.predict_next_tag(word, prev_tag, words, i)
            tags.append(next_tag)
            prev_tag = next_tag
            
        return tags
    
    def train(self, sequences: list, n_iter: int = 5) -> None:
        """Train the structured perceptron
        
        Args:
            sequences: List of (words, tags) sequences
            n_iter: Number of iterations
        """
        # Initialize weights to zero
        for epoch in range(n_iter):
            correct_sequences = 0
            total_sequences = 0
            
            for words, gold_tags in sequences:
                total_sequences += 1
                
                # Get predicted tag sequence using current model
                pred_tags = self.predict_sequence(words)
                
                # Check if the entire sequence was correct
                if pred_tags == gold_tags:
                    correct_sequences += 1
                    continue
                
                # Update weights based on the difference between predicted and gold
                prev_gold = "O"
                prev_pred = "O"
                
                for i, (word, gold_tag, pred_tag) in enumerate(zip(words, gold_tags, pred_tags)):
                    # Extract features for gold and predicted tags
                    gold_features = self.get_features(word, gold_tag, prev_gold, words, i)
                    pred_features = self.get_features(word, pred_tag, prev_pred, words, i)
                    
                    # Update weights: increase for gold features, decrease for predicted
                    for feature in gold_features:
                        self.theta[feature] += 1
                        
                    for feature in pred_features:
                        self.theta[feature] -= 1
                    
                    # Update previous tags
                    prev_gold = gold_tag
                    prev_pred = pred_tag
            
            # Print progress
            accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0
            print(f"Epoch {epoch+1}/{n_iter}, Sequence Accuracy: {accuracy:.4f}")
