"""wvv.py - Word Vector Viewer for GloVe format"""
import sys
import gzip
import heapq
from math import isnan, sqrt
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

class Vector:
    """Represents a single word vector"""

    def __init__(self, word: str, vector_values: list[float]) -> None:
        """Creates a Vector object from word and vector values"""
        self.word = word
        self.vector = vector_values
        self.magnitude = sqrt(sum(i**2 for i in self.vector))

    def __truediv__(self, k: float) -> "Vector":
        """Returns a new Vector object, whose elements are those of self divided by k."""
        vec_div = [i/k for i in self.vector]
        return Vector(self.word + "_div", vec_div)

    def __add__(self, vector: "Vector") -> "Vector":
        """Returns new Vector containing the sum of two vectors."""
        vec_sum = [i+j for i, j in zip(self.vector, vector.vector)]
        return Vector(self.word + "_sum", vec_sum)
    
    def __repr__(self):
        """Returns a string representation of the vector"""
        return f"Vector({self.word}, {len(self.vector)} dims)"

    def cosine_similarity(self, vector2: "Vector") -> float:
        """Returns cosine similarity between this vector and another vector."""
        if self.magnitude == 0 or vector2.magnitude == 0:
            return 0.0
        dot_product = sum(i*j for i, j in zip(self.vector, vector2.vector))
        return dot_product / (self.magnitude * vector2.magnitude)

    def cosine_distance(self, vector2: "Vector") -> float:
        """Returns cosine distance (1 - cosine similarity)"""
        return 1.0 - self.cosine_similarity(vector2)


class Vectorspace:
    """Represents a vectorspace containing word vectors"""

    def __init__(self, filename: str) -> None:
        """Creates a vectorspace object from GloVe format file"""
        self.filename = filename
        self.nr_words = None
        self.dimensionality = None
        self.vocabulary = {}  # key: word, value: Vector object
        self.load_vectors()

    def __getitem__(self, word: str) -> Vector:
        """Returns vector for given word, raises KeyError if not found"""
        if word not in self.vocabulary:
            raise KeyError(f"Word '{word}' not found in vocabulary")
        return self.vocabulary[word]

    def load_vectors(self) -> None:
        """Loads vectors from GloVe format file"""
        open_func = gzip.open if self.filename.endswith('.gz') else open
        
        with open_func(self.filename, 'rt', encoding='utf-8') as file:
            # Read first line to get vocabulary size and dimensionality
            first_line = file.readline().strip().split()
            self.nr_words = int(first_line[0])
            self.dimensionality = int(first_line[1])
            
            print(f"Loading {self.nr_words} words with {self.dimensionality} dimensions...")
            
            # Read remaining lines
            for line_num, line in enumerate(file, 2):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split()
                    word = parts[0]
                    vector_values = [float(x) for x in parts[1:]]
                    
                    if len(vector_values) != self.dimensionality:
                        print(f"Warning: Line {line_num} has {len(vector_values)} values, expected {self.dimensionality}")
                        continue
                    
                    vector = Vector(word, vector_values)
                    self.vocabulary[word] = vector
                    
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
        
        print(f"Vocabulary created with {len(self.vocabulary)} words from {self.filename}")

    def view_vector(self, word: str) -> None:
        """Prints the vector values for a given word"""
        try:
            vector = self[word]
            print(f"Vector for '{word}':")
            print(f"Dimensions: {len(vector.vector)}")
            print(f"Values: {vector.vector}")
            print(f"Magnitude: {vector.magnitude:.6f}")
        except KeyError:
            print(f"Word '{word}' not found in the vectorspace.")
            self.suggest_similar_words(word)

    def suggest_similar_words(self, word: str, max_suggestions: int = 5) -> None:
        """Suggests words that are similar to the input word (simple string matching)"""
        suggestions = []
        word_lower = word.lower()
        
        for vocab_word in self.vocabulary.keys():
            if word_lower in vocab_word.lower() or vocab_word.lower().startswith(word_lower[:3]):
                suggestions.append(vocab_word)
                if len(suggestions) >= max_suggestions:
                    break
        
        if suggestions:
            print(f"Did you mean one of these? {', '.join(suggestions)}")

    def find_similar_words(self, word: str, n: int = 10) -> list[tuple[str, float]]:
        """Finds the n most similar words to the given word using cosine similarity"""
        try:
            target_vector = self[word]
        except KeyError:
            print(f"Word '{word}' not found in vocabulary.")
            return []
        
        similarities = []
        
        for other_word, other_vector in self.vocabulary.items():
            if other_word == word:  # Skip the word itself
                continue
            
            similarity = target_vector.cosine_similarity(other_vector)
            if not isnan(similarity):
                # Use negative similarity for max-heap behavior with heapq (min-heap)
                heapq.heappush(similarities, (-similarity, other_word))
                
                # Keep only top n words
                if len(similarities) > n:
                    heapq.heappop(similarities)
        
        # Convert back to positive similarities and sort
        result = [(-sim, word) for sim, word in similarities]
        result.sort(reverse=True)
        
        return [(word, sim) for sim, word in result]

    def print_similar_words(self, word: str, n: int = 10) -> None:
        """Prints the n most similar words to the given word"""
        similar_words = self.find_similar_words(word, n)
        
        if not similar_words:
            return
        
        print(f"\nTop {len(similar_words)} words most similar to '{word}':")
        print("-" * 50)
        for i, (similar_word, similarity) in enumerate(similar_words, 1):
            print(f"{i:2d}. {similar_word:<20} (similarity: {similarity:.4f})")

    def contains_word(self, word: str) -> bool:
        """Check if word exists in vocabulary"""
        return word in self.vocabulary


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 wvv.py <vector_file> <word> [num_similar_words]")
        print("Example: python3 wvv.py glove.6B.100d.txt king 10")
        sys.exit(1)
    
    vector_file = sys.argv[1]
    query_word = sys.argv[2]
    num_similar = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    try:
        print("Loading word vectors...")
        vectorspace = Vectorspace(vector_file)
        
        # View the specific word vector
        vectorspace.view_vector(query_word)
        
        # Find and print similar words
        vectorspace.print_similar_words(query_word, num_similar)
        
    except FileNotFoundError:
        print(f"Error: File '{vector_file}' not found.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()