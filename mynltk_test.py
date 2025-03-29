import mynltk
from mynltk import Tokenizer, CategorizedCorpus

DATA_DIR = "data"

# Test the Tokenizer class
a_text = "This is a test. This is only a test. 1 2 3 4"

tokenizer = mynltk.Tokenizer()
tokens = tokenizer.simple_tokenize(a_text)
tokens2 = tokenizer.tokenize(a_text)
char_tokens = tokenizer.tokenize(a_text, char_based=True)
print(tokens)
print(tokens2)
print(char_tokens)

# Test the CategorizedCorpus class

cc = CategorizedCorpus("data/imdb/train")
for text, category in cc:
    print(f"{text[:40]}... {category}")