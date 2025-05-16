"""
Use pyTorch to implement the original neural language model of Bengio
et al. (2003), It is sufficient to implement
the following configuration:
    • Use word embeddings (referred to in the paper as “word features”) of
    size 100.
    • Use a hidden layer of size 80, with tanh activation.
    • Use a context of the 4 previous words to predict the following one,
        equivalent to a 5-gram language model.
    • Include only words that occur at least 5 times in english-train.txt.gz.

Depending on tokenization, this should give you a vocabulary of about
31000 words. All other tokens should be replaced by a special <UNK> token for
unknown words.
mv
• As with the n-gram models, use a special <PAD> token to fill out before
the start of the sentence in order to always get a 4-word context. Also
add a special end-of-sentence <EOS> token at the end of each sentence.

• Use gradient clipping with a threshold of 1.0.
Train for two epochs on english-train.txt.gz, using the Adam optimizer
(not available when Bengio et al. (2003) created their model – but it makes
the training faster) with a learning rate of 0.001 and a batch size of 1024.
Exclude examples where <UNK> is the target word, but leave those where it
occurs in the 4-word context.
Write the embeddings to a file of the format used in Section 2.3, so
that you can easily perform a sanity check. For instance, the 10 words most
similar to “Friday” should look something like this (with cosine similarities)
"""
