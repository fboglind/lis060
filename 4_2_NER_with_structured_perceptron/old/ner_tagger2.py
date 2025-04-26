"""ner_tagger.py"""
from structured_perceptron import StructuredPerceptron


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
        
        # Previous tag feature
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

