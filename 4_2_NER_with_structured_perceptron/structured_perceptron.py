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
            

class NERTagger:
    """A class for NER tagging using the structured perceptron"""
    
    def __init__(self, n_iters: int = 5) -> None:
        """Constructs a NERTagger object
        
        Args:
            n_iters: Number of training iterations
        """
        self.n_iters = n_iters
        self.model = None
        
    def get_features(self, word: str, tag: str, prev_tag: str = None, 
                     sentence=None, position=None) -> set:
        """Extract features for NER
        
        Args:
            word: The current word
            tag: The NER tag
            prev_tag: Previous tag in sequence
            sentence: Full sentence
            position: Position in sentence
            
        Returns:
            Set of feature strings
        """
        features = set()
        
        # Word features
        features.add(f"word.form={word}")
        features.add(f"word.lower={word.lower()}")
        features.add(f"word.is_lower={word.islower()}")
        features.add(f"word.is_upper={word.isupper()}")
        features.add(f"word.is_title={word.istitle()}")
        features.add(f"word.is_digit={word.isdigit()}")
        features.add(f"word.is_alpha={word.isalpha()}")
        
        # Previous tag feature - essential for structured prediction
        if prev_tag:
            features.add(f"prev_tag={prev_tag}")
            features.add(f"prev_tag={prev_tag}+word.form={word}")
            features.add(f"prev_tag={prev_tag}+word.lower={word.lower()}")
        
        # Include prefix/suffix features
        for i in range(1, 5):
            if len(word) >= i:
                prefix = word[:i]
                suffix = word[-i:]
                features.add(f"prefix{i}={prefix}")
                features.add(f"suffix{i}={suffix}")
        
        # Context features if sentence is provided
        if sentence and position is not None:
            # Previous word features
            if position > 0:
                prev_word = sentence[position-1]
                features.add(f"prev_word={prev_word}")
                features.add(f"prev_word.lower={prev_word.lower()}")
                features.add(f"prev_word.is_title={prev_word.istitle()}")
            else:
                features.add("BOS")  # Beginning of sentence
                
            # Next word features
            if position < len(sentence) - 1:
                next_word = sentence[position+1]
                features.add(f"next_word={next_word}")
                features.add(f"next_word.lower={next_word.lower()}")
                features.add(f"next_word.is_title={next_word.istitle()}")
            else:
                features.add("EOS")  # End of sentence
        
        # Add label to all features
        labeled_features = set()
        for feat in features:
            labeled_features.add(f"{feat}+label={tag}")
        
        return labeled_features
    
    def train(self, training_data: list) -> None:
        """Train the NER tagger
        
        Args:
            training_data: List of (sentence, tags) tuples
        """
        # Extract all possible NER tags from the training data
        all_tags = set()
        for _, tags in training_data:
            all_tags.update(tags)
        
        # Initialize the structured perceptron
        self.model = StructuredPerceptron(all_tags, self.get_features)
        
        # Train the model
        self.model.train(training_data, self.n_iters)
        
    def predict(self, sentences: list) -> list:
        """Predict NER tags for sentences
        
        Args:
            sentences: List of sentences (each a list of words)
            
        Returns:
            List of tag sequences
        """
        if not self.model:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = []
        for sentence in sentences:
            tags = self.model.predict_sequence(sentence)
            predictions.append(tags)
            
        return predictions
    
    def evaluate(self, test_data: list) -> dict:
        """Evaluate the model on test data
        
        Args:
            test_data: List of (sentence, gold_tags) tuples
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.model:
            raise ValueError("Model not trained. Call train() first.")
        
        correct_tokens = 0
        total_tokens = 0
        correct_sequences = 0
        total_sequences = 0
        
        for sentence, gold_tags in test_data:
            total_sequences += 1
            pred_tags = self.model.predict_sequence(sentence)
            
            # Check token accuracy
            total_tokens += len(sentence)
            correct_tokens += sum(1 for g, p in zip(gold_tags, pred_tags) if g == p)
            
            # Check sequence accuracy
            if gold_tags == pred_tags:
                correct_sequences += 1
        
        return {
            "token_accuracy": correct_tokens / total_tokens if total_tokens > 0 else 0,
            "sequence_accuracy": correct_sequences / total_sequences if total_sequences > 0 else 0
        }
    
    def show_top_features(self, n: int = 10) -> None:
        """Print top features with highest weights
        
        Args:
            n: Number of top features to show
        """
        if not self.model:
            raise ValueError("Model not trained. Call train() first.")
            
        # Get features sorted by weight
        top_features = sorted(
            self.model.theta.items(), key=lambda x: abs(x[1]), reverse=True
        )[:n]
        
        print(f"Top {n} features:")
        for idx, (feature, weight) in enumerate(top_features, start=1):
            print(f"{idx}. {feature}: {weight:.4f}")

