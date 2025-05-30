# tests/test_neural_ngram.py
import torch
from torch.utils.data import DataLoader
from mynltk.neural_ngram import NeuralNGramLM, NeuralNGramDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    dataset = NeuralNGramDataset("data/bengio/english-train.txt.gz")
    vocab_size = len(dataset.vocab)
    model = NeuralNGramLM(vocab_size).to(device)

    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    criterion = torch.nn.NLLLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        total_loss = 0
        total_tokens = 0
        for contexts, targets in dataloader:
            contexts, targets = contexts.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(contexts)
            loss = criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_tokens += len(targets)

        avg_loss = total_loss / total_tokens
        print(f"Epoch {epoch+1} | Avg. loss (NLL): {avg_loss:.4f}")
    
        # Export embeddings
    output_path = "./neural_ngram.vec.gz"
    model.export_embeddings(output_path, dataset.itos)
    print(f"Embeddings exported to {output_path}")


if __name__ == "__main__":
    train()
