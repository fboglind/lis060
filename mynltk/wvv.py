"""wvv.py"""
import sys
import gzip
import heapq

class Vector:
    """Represents a single vector (in a vectorspace)

    Note: the class contains two distance_methods, one measuring cosine-distance, and one measuring
    euclidian-distance(edited out). The user may choose which measure to use.
    """

    def __init__(self, row: list["Vector"]) -> None:
        """Creates a Vector-object"""
        self.word=row.split(" ", 1)[0]
        self.vector=[float(i) for i in row]
        self.magnitude=(sum(i**2 for i in self.vector))**(1/2)

    def __truediv__(self, k: int) -> "Vector":
        """Returns a new Vector object, whose elements are those of self divided by k."""
        vec_div=[i/k for i in self.vector]
        out_vec=Vector(vec_div)
        return out_vec

    def __add__(self, vector: "Vector") -> "Vector":
        """Returns new Vector containing the sum of two vectors (self and vector in argument)."""
        vec_sum= [i+j for i, j in zip(self.vector, vector.vector)]
        out_vec=Vector(vec_sum)
        return out_vec
    
    def __repr__(self):
        """Returns a string representation of the vector"""
        return f"Vector({self.word}, {self.vector})"

    def distance(self, vector2: "Vector") -> float:
        """Returns cosine-distance between a vector and another vector."""
        return float(1.0-(self.calculate_dot_product(vector2)/self.magnitude * vector2.magnitude))

    def calculate_dot_product(self, vector2: "Vector") -> float:
        """Calculates the dot product of a vector and another vector"""
        dot_product=sum(i*j for i, j in zip(self.vector, vector2.vector))
        return dot_product

class Vectorspace():
    """Represents a vectorspace containing vectors"""

    def __init__(self, filename: str) -> None:
        """Creates a vectorspace-object"""
        self.filename1=filename
        self.nr_words=None
        self.dimensionality=None
        with open (self.filename, encoding="utf-8") as file:
            first_line=file.readline()
            self.nr_words=first_line[0]
            self.dimensionality=first_line[1]
        self.vocabulary = {} #key:word, value:vector-object
        self.create_vocabulary()

    def __getitem__(self, word: str) -> Vector:
        """Takes a word and returns a vector if it exists in the vectorspace, if not an error-
        message is printed"""
        vector= self.vocabulary.get(word)
        if vector is None:
            raise KeyError
        return vector

    def create_vocabulary(self) -> None:
        """Reads a file creates vectors and adds words and vector-objects to a vocabulary"""
        with open (self.filename, encoding="utf-8") as file:
            file.readline() # Skip the first line
            for line in file:
                line = line.strip()
                if not line:
                    continue
                # Create a Vector object and add it to the vocabulary
                vector=Vector(line)
            self.vocabulary[vector.word]=vector
        print(f"Vocabulary created from {self.filename}.")

    def view_vector(self, word: str) -> None:
        """Prints the vector of a word"""
        try:
            vector=self.vectorspace[word]
            print(f"Vector for {word}: {vector.vector}")
        except KeyError:
            print(f"{word} not found in the vectorspace.")


class VectorViwewer():
    def get_similar_words(word: str, vs1: Vectorspace, vs2: Vectorspace) -> None:
    """Takes an input-word and two vectorspaces and prints the five most similar words in terms of
    cosine distane to the input-word for each vectorspace
    """
    word_vector= vs1[word]
    l1_heap=[]#[tuple(float, str)]
    l2_heap=[]#[tuple(float, str)]
    for vs1_vec, vs2_vec in zip(vs1.vocabulary.values(), vs2.vocabulary.values()):
        cos_dis1=word_vector.distance(vs1_vec)
        cos_dis2=word_vector.distance(vs2_vec)
        if not isnan(cos_dis1):
            make_heap(cos_dis1, vs1_vec.word, l1_heap)
        if not isnan(cos_dis2):
            make_heap(cos_dis2, vs2_vec.word, l2_heap)
    print_list(l1_heap)
    print_list(l2_heap)

def make_heap(cos_dis: float, word: str, a_heap: list) -> None:
    """Negates a value and pushes it onto a heap, if the heap contains five values the smallest
    value is popped.
    """
    heapq.heappush(a_heap, tuple ((-cos_dis, word)))
    if len(a_heap) > 5:
        heapq.heappop(a_heap)

def print_list(a_heap: list) -> None:
    """Prints the ten most similar words in a list"""
    print("Most similar words:")
    for i in range(10):
        value, word=heapq.heappop(a_heap)
        print(f"{word}: {-value}")
    print("\n")


def main():
    path = sys.argv[1]
    word= sys.argv[2]


    with gzip.open(path, "rb") as f:
        vecspace=Vectorspace(f)
        vecspace.view_vector(word)

    for vector in vecspace.vocabulary.values():
        vector.distance(vecspace[word])


if __name__=="__main__":
    main()










