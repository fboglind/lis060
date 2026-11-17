# mynltk/neural_ngram.py
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
    """Neural n-gram language model using embeddings and feedforward neural network."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 100, hidden_size: int = 80, context_size: int = 4):
        """Initialize the neural n-gram language model."""
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights properly
        nn.init.uniform_(self.embeddings.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        embeds = self.embeddings(x)  # (batch_size, context_size, embedding_dim)
        embeds = embeds.view(embeds.size(0), -1)  # Flattens
        hidden = torch.tanh(self.linear1(embeds))
        output = self.linear2(hidden)
        return F.log_softmax(output, dim=1)
    
    def export_embeddings(self, filepath: str, itos: dict):
        """Export embeddings in GloVe format."""
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            num_words = len(itos)
            dim = self.embeddings.embedding_dim
            f.write(f"{num_words} {dim}\n")
            
            with torch.no_grad():
                for idx, word in itos.items():
                    vec = self.embeddings.weight[idx].cpu().numpy()
                    vec_str = " ".join(f"{x:.6f}" for x in vec)
                    f.write(f"{word} {vec_str}\n")


class NeuralNGramDataset(Dataset):
    def __init__(self, filepath: str, min_freq: int = 5, context_size: int = 4):
        """Initialize the dataset for neural n-gram language modeling."""
        self.context_size = context_size
        self.min_freq = min_freq
        self.token_counts = Counter()
        self.sentences = []

        # First pass: count tokens
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:  # Skip empty lines
                    tokens.append(EOS_TOKEN)  # Add EOS at the end
                    self.token_counts.update(tokens)
                    self.sentences.append(tokens)

        # Build vocabulary with special tokens first
        special_tokens = [PAD_TOKEN, UNK_TOKEN, EOS_TOKEN]
        self.vocab = {token: i for i, token in enumerate(special_tokens)}
        
        # Add frequent tokens
        idx = len(special_tokens)
        for token, count in self.token_counts.most_common():
            if token not in self.vocab and count >= min_freq:
                self.vocab[token] = idx
                idx += 1
        
        self.vocab_size = len(self.vocab)
        self.itos = {i: tok for tok, i in self.vocab.items()}

        # Build training examples
        self.data = []
        for tokens in self.sentences:
            # Add padding at the beginning
            padded = [PAD_TOKEN] * context_size + tokens
            
            for i in range(context_size, len(padded)):
                context = padded[i - context_size:i]
                target = padded[i]
                
                # Skip if target is not in vocabulary (would be UNK)
                if target not in self.vocab:
                    continue
                
                context_ids = [self.vocab.get(tok, self.vocab[UNK_TOKEN]) for tok in context]
                target_id = self.vocab[target]
                self.data.append((context_ids, target_id))

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        context_ids, target_id = self.data[idx]
        return torch.tensor(context_ids, dtype=torch.long), torch.tensor(target_id, dtype=torch.long)