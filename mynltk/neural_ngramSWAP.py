# mynltk/neural_gram.py
import gzip
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset, DataLoader

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"

class NeuralNGramLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_size=80, context_size=4):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embeds = self.embeddings(x)  # (batch_size, context_size, embedding_dim)
        embeds = embeds.view(embeds.size(0), -1)  # Flatten
        hidden = torch.tanh(self.linear1(embeds))
        output = F.log_softmax(self.linear2(hidden), dim=1)
        return output


class NeuralNGramDataset(Dataset):
    def __init__(self, filepath, min_freq=5, context_size=4):
        self.context_size = context_size
        self.token_counts = Counter()
        self.sentences = []

        # First pass: build vocabulary
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split() + [EOS_TOKEN]
                self.token_counts.update(tokens)
                self.sentences.append(tokens)

        self.vocab = {
            token: i for i, (token, count) in enumerate(
                [(PAD_TOKEN, math.inf), (UNK_TOKEN, math.inf), (EOS_TOKEN, math.inf)] +
                [(t, c) for t, c in self.token_counts.items() if c >= min_freq and t not in {EOS_TOKEN}]
            )
        }
        self.itos = {i: tok for tok, i in self.vocab.items()}

        self.data = []
        for tokens in self.sentences:
            tokens = [PAD_TOKEN] * context_size + tokens
            for i in range(context_size, len(tokens)):
                context = tokens[i - context_size:i]
                target = tokens[i]
                if target not in self.vocab:
                    continue  # Exclude if target is <UNK>
                context_ids = [self.vocab.get(tok, self.vocab[UNK_TOKEN]) for tok in context]
                target_id = self.vocab[target]
                self.data.append((context_ids, target_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context_ids, target_id = self.data[idx]
        return torch.tensor(context_ids, dtype=torch.long), torch.tensor(target_id, dtype=torch.long)
