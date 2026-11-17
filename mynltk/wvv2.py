"""
Word Vector Viewer using sklearn cosine_similarity
Clean implementation for comparison with custom code
"""
import sys
import gzip
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class WordVectors:
    def __init__(self, filepath):
        self.filepath = filepath
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vectors = None
        self.vocab_size = 0
        self.dimensions = 0
        self.load_vectors()
    
    def load_vectors(self):
        """Load GloVe format vectors into numpy array"""
        open_func = gzip.open if self.filepath.endswith('.gz') else open
        
        with open_func(self.filepath, 'rt', encoding='utf-8') as f:
            # Read header
            first_line = f.readline().strip().split()
            self.vocab_size = int(first_line[0])
            self.dimensions = int(first_line[1])
            
            print(f"Loading {self.vocab_size} words with {self.dimensions} dimensions...")
            
            # Initialize arrays
            self.vectors = np.zeros((self.vocab_size, self.dimensions))
            words = []
            
            # Read vectors
            for idx, line in enumerate(f):
                if idx >= self.vocab_size:
                    break
                    
                parts = line.strip().split()
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                
                self.vectors[idx] = vector
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                words.append(word)
        
        print(f"Loaded {len(words)} word vectors")
    
    def get_vector(self, word):
        """Get vector for a specific word"""
        if word not in self.word_to_idx:
            raise KeyError(f"Word '{word}' not found in vocabulary")
        idx = self.word_to_idx[word]
        return self.vectors[idx]
    
    def view_vector(self, word):
        """Display vector information for a word"""
        try:
            vector = self.get_vector(word)
            print(f"\nVector for '{word}':")
            print(f"Dimensions: {len(vector)}")
            print(f"Values: {vector}")
            print(f"Magnitude: {np.linalg.norm(vector):.6f}")
        except KeyError:
            print(f"Word '{word}' not found in vocabulary")
            self.suggest_words(word)
    
    def suggest_words(self, word, max_suggestions=5):
        """Suggest similar words based on string matching"""
        suggestions = [w for w in self.word_to_idx.keys() 
                      if word.lower() in w.lower() or w.lower().startswith(word.lower()[:3])]
        if suggestions[:max_suggestions]:
            print(f"Did you mean: {', '.join(suggestions[:max_suggestions])}")
    
    def find_similar_words(self, word, n=10):
        """Find n most similar words using sklearn cosine_similarity"""
        try:
            target_vector = self.get_vector(word).reshape(1, -1)
        except KeyError:
            print(f"Word '{word}' not found")
            return []
        
        # Calculate cosine similarities with all vectors
        similarities = cosine_similarity(target_vector, self.vectors)[0]
        
        # Get indices of most similar words (excluding the word itself)
        target_idx = self.word_to_idx[word]
        similarities[target_idx] = -1  # Exclude self
        
        # Get top n similar words
        top_indices = np.argsort(similarities)[::-1][:n]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > -1:  # Skip if it was the original word
                similar_word = self.idx_to_word[idx]
                similarity_score = similarities[idx]
                results.append((similar_word, similarity_score))
        
        return results
    
    def print_similar_words(self, word, n=10):
        """Print the n most similar words"""
        similar_words = self.find_similar_words(word, n)
        
        if not similar_words:
            return
        
        print(f"\nTop {len(similar_words)} words most similar to '{word}':")
        print("-" * 50)
        for i, (similar_word, similarity) in enumerate(similar_words, 1):
            print(f"{i:2d}. {similar_word:<20} (similarity: {similarity:.4f})")
    
    def contains_word(self, word):
        """Check if word exists in vocabulary"""
        return word in self.word_to_idx
    
    def get_vocab_stats(self):
        """Print vocabulary statistics"""
        print(f"\nVocabulary Statistics:")
        print(f"Total words: {self.vocab_size}")
        print(f"Vector dimensions: {self.dimensions}")
        print(f"Memory usage: {self.vectors.nbytes / 1024 / 1024:.1f} MB")


def main():
    if len(sys.argv) < 3:
        print("Usage: python sklearn_wv.py <vector_file> <word> [num_similar]")
        print("Example: python sklearn_wv.py glove.6B.100d.txt king 10")
        sys.exit(1)
    
    vector_file = sys.argv[1]
    query_word = sys.argv[2]
    num_similar = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    try:
        # Load vectors
        wv = WordVectors(vector_file)
        wv.get_vocab_stats()
        
        # View specific word vector
        wv.view_vector(query_word)
        
        # Find and display similar words
        wv.print_similar_words(query_word, num_similar)
        
    except FileNotFoundError:
        print(f"Error: File '{vector_file}' not found")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()