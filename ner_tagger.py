"""ner_tagger.py"""
from structured_perceptron import StructuredPerceptron


class NERTagger:
    """A class for NER tagging using the structured perceptron
    Attributes:
        - n_iters: Number of training iterations
        - model: The structured perceptron model"""
    
    def __init__(self, n_iters: int = 5) -> None:
        """Constructs a NERTagger object
        
        Args:
            n_iters: Number of training iterations
        """
        self.n_iters = n_iters
        self.model = None

    def get_features(self, word: str, tag: str, prev_tag: str = None, 
                 sentence=None, position=None) -> set[str]:
        """Extract features for a word in a sentence
        Args:
            word: The word to extract features from
            tag: The NER tag for the word
            prev_tag: The previous tag in the sequence
            sentence: The sentence containing the word
            position: The position of the word in the sentence
        Returns:
            A set of features for the word    
        """
        features = set()
        
        features |= get_word_shape_features(word)
        features |= get_affix_features(word)
        features |= get_prev_tag_features(prev_tag, word)
        
        if sentence is not None and position is not None:
            features |= get_context_features(sentence, position)

        # Add the label to all features
        return {f"{feature}+label={tag}" for feature in features}


    
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
            
            # Check TOKEN ACCURACY = Total nr of tokens/Nr of correctly tagged tokens
            total_tokens += len(sentence)
            correct_tokens += sum(1 for g, p in zip(gold_tags, pred_tags) if g == p)
            
            # Check SEQUENCE ACCURACY = Total nr of sequences/Nr of fully correct sequences
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

# Helper functions to extract features

def get_word_shape_features(word: str) -> set[str]:
        """Extract word shape features
        Args:
            word: The word to extract features from
        Returns:
            A set of word shape features
        """
        return {
            f"word.form={word}",
            f"word.lower={word.lower()}",
            f"word.is_lower={word.islower()}",
            f"word.is_upper={word.isupper()}",
            f"word.is_title={word.istitle()}",
            f"word.is_digit={word.isdigit()}",
            f"word.is_alpha={word.isalpha()}",
        }
    
def get_affix_features(word: str, max_len: int = 4) -> set[str]:
    """Extract affix features
    Args:
        word: The word to extract affix features from
        max_len: The maximum length of affixes to consider
    Returns:
        A set of affix features"""
    features = set()
    for i in range(1, max_len + 1):
        if len(word) >= i:
            features.add(f"prefix{i}={word[:i]}")
            features.add(f"suffix{i}={word[-i:]}")
    return features

def get_prev_tag_features(prev_tag: str, word: str) -> set[str]:
    """Extract previous tag features
    Args:
        prev_tag: The previous tag in the sequence
        word: The current word
        Returns:
            A set of previous tag features"""
    if not prev_tag:
        return set()
    return {
        f"prev_tag={prev_tag}",
        f"prev_tag={prev_tag}+word.form={word}",
        f"prev_tag={prev_tag}+word.lower={word.lower()}",
    }

def get_context_features(sentence: list[str], position: int) -> set[str]:
    """Extract features based on context (previous and next words)
    Args:
        sentence: The sentence containing the word
        position: The position of the word in the sentence
    Returns:
        A set of context features
    """
    features = set()
    if position > 0:
        prev_word = sentence[position - 1]
        features.add(f"prev_word={prev_word}")
        features.add(f"prev_word.lower={prev_word.lower()}")
        features.add(f"prev_word.is_title={prev_word.istitle()}")
    else:
        features.add("BOS")

    if position < len(sentence) - 1:
        next_word = sentence[position + 1]
        features.add(f"next_word={next_word}")
        features.add(f"next_word.lower={next_word.lower()}")
        features.add(f"next_word.is_title={next_word.istitle()}")
    else:
        features.add("EOS")
    return features
