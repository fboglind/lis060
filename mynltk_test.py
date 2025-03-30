import tokenizer#, MorphologicalAnalyzer#, CategorizedCorpus


DATA_DIR = "data"

# Test the Tokenizer class
a_text = "This is a test. This is only a test. 1 2 3 4"

tokenizer = tokenizer.Tokenizer()
tokens = tokenizer.simple_tokenize(a_text)
tokens2 = tokenizer.tokenize(a_text)
char_tokens = tokenizer.tokenize(a_text, char_based=True)
print(tokens)
print(tokens2)
print(char_tokens)

# Test the CategorizedCorpus class

# cc = CategorizedCorpus("data/imdb/train")
# for text, category in cc:
#     print(f"{text[:40]}... {category}")

# Test Perceptron and Analyzer classes
def read_unimorph(filepath: str, testing=False, use_surface_form=False) -> list:
    """Read UniMorph data and prepare for feature-based analysis
        Args:
            filepath: Path to datafile (str)
            testing: Use this for the test set (bool)
            use_surface_form: False allows for multiword expressions (bool)
        Returns:
            data: list of words and tags (list[tuple])
            """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            base, inflected, tags = line.strip().split('\t')
            # Handle multi-word expressions
            inflected = inflected.replace(" ", "_") if use_surface_form else inflected
            pos = tags.split(';')[0]
            if testing:
                data.append((inflected, pos))  # Just the input
            else:
                data.append(((inflected, pos), tags))  # ((form, POS), full tag)
    return data

def compute_accuracy(gold_data, predicted_tags):
    """Computes accuracy by comparing test and gold sets"""
    correct = sum(1 for ((_, _), gold) , pred in zip(gold_data, predicted_tags) if gold == pred)
    return correct / len(gold_data)


# Load data
train_data = read_unimorph("data/unimorph/swe-train")  # Returns [(base, inflected, tags)]
test_data = read_unimorph("data/unimorph/swe-test", testing=True)  # Returns [inflected_words]
gold_data =read_unimorph("data/unimorph/swe-test", testing=False)

print(test_data[4])
print(test_data[6])
print(test_data[7])
print(gold_data[4])
# Train and analyze
analyzer = MorphologicalAnalyzer(training_data=train_data, n_iters=3)
# Predict
predictions = analyzer.predict(test_data)
# Show top features
analyzer.show_top_features()

accuracy=compute_accuracy(gold_data,predictions)
print(f"Accuracy: {accuracy:.2%}")

#Test BPE encoder
def whitespace_separate(C:str)->str:
    """Separates characters in a string with whitespace and adds end of word symbol: '_'"""
    C.replace(" ", "_")
    c_list=list(C)
    return str(c_list)

text="This is a text"

print(whitespace_separate(text))
