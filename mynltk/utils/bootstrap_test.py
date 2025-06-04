"""bootstrap_test.py
Runs paired bootstrap significance test (as described by Jurafsky & Martin) on saved predictions.

Usage:
    python bootstrap_test.py gold.txt system1.txt system2.txt

Each file should contain one label per line, corresponding to the same instance.
"""
import sys
import random


def read_labels(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def accuracy(gold, pred):
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)

def paired_bootstrap(gold, sys1, sys2, num_samples=1000, seed=42):
    random.seed(seed)
    n = len(gold)
    delta = accuracy(gold, sys1) - accuracy(gold, sys2)
    s = 0
    for _ in range(num_samples):
        indices = [random.randint(0, n - 1) for _ in range(n)]
        gold_sample = [gold[i] for i in indices]
        sys1_sample = [sys1[i] for i in indices]
        sys2_sample = [sys2[i] for i in indices]
        delta_i = accuracy(gold_sample, sys1_sample) - accuracy(gold_sample, sys2_sample)
        if abs(delta_i) >= abs(2 * delta):
            s += 1
    return delta, s / num_samples

# def main():
#     if len(sys.argv) != 4:
#         print("Usage: python bootstrap_test.py gold.txt system1.txt system2.txt")
#         sys.exit(1)

#     gold_path, sys1_path, sys2_path = sys.argv[1:]
#     gold = read_labels(gold_path)
#     sys1 = read_labels(sys1_path)
#     sys2 = read_labels(sys2_path)

#     assert len(gold) == len(sys1) == len(sys2), "All files must be same length."

#     delta, p_value = paired_bootstrap(gold, sys1, sys2)
#     print(f"Δ (accuracy difference): {delta:.4f}")
#     print(f"P-value: {p_value:.4f}")
#     if p_value < 0.05:
#         print("→ The difference is statistically significant.")
#     else:
#         print("→ No statistically significant difference detected.")

# if __name__ == "__main__":
#     main()