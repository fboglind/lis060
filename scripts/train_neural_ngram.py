#!/usr/bin/env python3
"""Train a neural n-gram language model following Bengio et al. (2003)"""

import gzip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import math
from tqdm import tqdm

# Special tokens
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
        
        # Model layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values"""
        # Initialize embeddings
        nn.init.uniform_(self.embeddings.weight, -0.1, 0.1)
        # Initialize linear layers
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        # x shape: (batch_size, context_size)
        embeds = self.embeddings(x)  # (batch_size, context_size, embedding_dim)
        embeds = embeds.view(embeds.size(0), -1)  # Flatten to (batch_size, context_size * embedding_dim)
        hidden = torch.tanh(self.linear1(embeds))  # (batch_size, hidden_size)
        output = self.linear2(hidden)  # (batch_size, vocab_size)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs
    
    def export_embeddings(self, filepath: str, itos: dict):
        """Export embeddings in GloVe format."""
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            num_words = len(itos)
            dim = self.embeddings.embedding_dim
            f.write(f"{num_words} {dim}\n")
            
            # Write embeddings for each word
            with torch.no_grad():
                for idx, word in itos.items():
                    vec = self.embeddings.weight[idx].cpu().numpy()
                    vec_str = " ".join(f"{x:.6f}" for x in vec)
                    f.write(f"{word} {vec_str}\n")


class NeuralNGramDataset(Dataset):
    """Dataset for neural n-gram language modeling."""
    
    def __init__(self, filepath: str, min_freq: int = 5, context_size: int = 4):
        """Initialize the dataset."""
        self.context_size = context_size
        self.min_freq = min_freq
        
        # First pass: count tokens
        print("Counting tokens...")
        self.token_counts = Counter()
        self.sentences = []
        
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading sentences"):
                tokens = line.strip().split() #Simplest possible tokenization, BPE?
                if tokens:  # Skip empty lines
                    # Add EOS token at the end
                    tokens.append(EOS_TOKEN)
                    self.token_counts.update(tokens)
                    self.sentences.append(tokens)
        
        # Build vocabulary (keeping special tokens regardless of frequency)
        print(f"Building vocabulary (min_freq={min_freq})...")
        special_tokens = [PAD_TOKEN, UNK_TOKEN, EOS_TOKEN]
        
        # Start with special tokens
        self.vocab = {token: i for i, token in enumerate(special_tokens)}
        
        # Add tokens that meet frequency threshold
        idx = len(special_tokens)
        for token, count in self.token_counts.most_common():
            if token not in self.vocab and count >= min_freq:
                self.vocab[token] = idx
                idx += 1
        
        self.vocab_size = len(self.vocab)
        self.itos = {i: tok for tok, i in self.vocab.items()}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Total sentences: {len(self.sentences)}")
        
        # Build training examples
        print("Building training examples...")
        self.data = []
        
        for sent in tqdm(self.sentences, desc="Processing sentences"):
            # Pad with context_size PAD tokens at the beginning
            padded_sent = [PAD_TOKEN] * context_size + sent
            
            # Create training examples
            for i in range(context_size, len(padded_sent)):
                context = padded_sent[i - context_size:i]
                target = padded_sent[i]
                
                # Skip if target is UNK (as specified in assignment)
                if target not in self.vocab:
                    continue
                
                # Convert to indices
                context_ids = [self.vocab.get(tok, self.vocab[UNK_TOKEN]) for tok in context]
                target_id = self.vocab[target]
                
                self.data.append((context_ids, target_id))
        
        print(f"Total training examples: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        context_ids, target_id = self.data[idx]
        return (torch.tensor(context_ids, dtype=torch.long), 
                torch.tensor(target_id, dtype=torch.long))


def train_model(model, train_loader, num_epochs=2, learning_rate=0.001, 
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train the neural n-gram model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(reduction='sum')  # Use sum for per-token loss calculation
    
    print(f"Training on {device}...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_tokens = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (contexts, targets) in enumerate(progress_bar):
            contexts = contexts.to(device)
            targets = targets.to(device)
            
            # Forward pass
            log_probs = model(contexts)
            loss = criterion(log_probs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Track statistics
            total_loss += loss.item()
            total_tokens += targets.size(0)
            
            # Update progress bar
            avg_loss = total_loss / total_tokens
            progress_bar.set_postfix({'loss/token': f'{avg_loss:.4f}'})
        
        print(f"Epoch {epoch+1} - Average loss per token: {total_loss/total_tokens:.4f}")


def evaluate_model(model, test_file, dataset, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Evaluate the model on a test file."""
    model = model.to(device)
    model.eval()
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        with gzip.open(test_file, "rt", encoding="utf-8") as f:
            for line in tqdm(f, desc="Evaluating"):
                tokens = line.strip().split()
                if not tokens:
                    continue
                
                # Add padding and EOS
                tokens = [PAD_TOKEN] * dataset.context_size + tokens + [EOS_TOKEN]
                
                # Convert to indices
                token_ids = [dataset.vocab.get(tok, dataset.vocab[UNK_TOKEN]) for tok in tokens]
                
                # Evaluate each position
                for i in range(dataset.context_size, len(token_ids)):
                    context = token_ids[i - dataset.context_size:i]
                    target = token_ids[i]
                    
                    # Skip unknown targets
                    if tokens[i] not in dataset.vocab:
                        continue
                    
                    # Prepare tensors
                    context_tensor = torch.tensor([context], dtype=torch.long).to(device)
                    target_tensor = torch.tensor([target], dtype=torch.long).to(device)
                    
                    # Get log probability
                    log_probs = model(context_tensor)
                    loss = -log_probs[0, target].item()  # Negative log likelihood
                    
                    total_loss += loss
                    total_tokens += 1
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    print(f"Test set - Average loss per token: {avg_loss:.4f}")
    print(f"Test set - Perplexity: {math.exp(avg_loss):.2f}")
    return avg_loss


def main():
    train_file = "data/bengio/english-train.txt.gz"
    test_file = "data/bengio/english-test-1k.txt.gz"
    
    embedding_dim = 100
    hidden_size = 80
    context_size = 4
    min_freq = 5
    batch_size = 1024
    num_epochs = 2
    learning_rate = 0.001
    
    # Create dataset
    print("Loading training data...")
    train_dataset = NeuralNGramDataset(train_file, min_freq=min_freq, context_size=context_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = NeuralNGramLM(
        vocab_size=train_dataset.vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        context_size=context_size
    )
    
    # Evaluate before training
    print("\nEvaluating before training...")
    initial_loss = evaluate_model(model, test_file, train_dataset)
    
    # Train model
    print("\nStarting training...")
    train_model(model, train_loader, num_epochs=num_epochs, learning_rate=learning_rate)
    
    # Evaluate after training
    print("\nEvaluating after training...")
    final_loss = evaluate_model(model, test_file, train_dataset)
    
    # Export embeddings
    print("\nExporting embeddings...")
    model.export_embeddings("neural_ngram.vec.gz", train_dataset.itos)
    print("Embeddings saved to neural_ngram.vec.gz")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': train_dataset.vocab,
        'itos': train_dataset.itos,
        'config': {
            'vocab_size': train_dataset.vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'context_size': context_size
        }
    }, 'neural_ngram_model.pt')
    print("Model saved to neural_ngram_model.pt")


if __name__ == "__main__":
    main()