"""conditional_language_model.py"""

"""A language model conditioned on an input vector. It is a very powerful tool,
since the input vector can be used to steer the generated text in a certain
direction. 
The conditioning vector could represent everything from a language in which to generate text, to a representation of an image to describe.
In this assignment, you will create an LSTM-based character language mod-
els which supports appending a low-dimensional feature vector to the input
(word embeddings) at each time step. One application of this is to select
the language in a multilingual language model. However, you do not need
to stick to a single language, but can request a mixture between languages,
or shift languages during the generation of a sentence.
In the directory lis060/data/charlm/europarl4-1M.txt.gz these is
one file that you can use for training, which contains a mix of character-
segmented sentences in four languages. Each line contains two tab-separated
fields: the first one contains feature values, such as swe=1.0 (the features
not mentioned are implicitly set to 0). The second field contains text, in the
form of space-separated one-character tokens, or underscore for space. You
can use this to train a character-based language model that takes language
identity is input. When generating, you can also try to mix languages, e.g.
deu=0.5;eng=0.5 for an even mix between German and English. It is also
possible to change the features during generation of a sentence, so that you
start with 100% English and gradually move to 100% German.
You can also easily create your own data, and experiment with for in-
stance different text genres.
The language model is very similar to a plain LSTM language model,
the only difference is that at each time step, you concatenate the embedding
"""