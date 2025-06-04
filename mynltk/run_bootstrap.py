"""run_bootstrap.py
Runs paired bootstrap significance testing (Jurafsky & Martin) on UniMorph test data,
using two prediction files and the shared read_unimorph utility. Saves results to a CSV.
"""

import sys
import csv
from pathlib import Path
from mynltk.utils.read_unimorph import read_unimorph
from mynltk.utils.bootstrap_test import paired_bootstrap

def read_labels_from_file(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    if len(sys.argv) != 4:
        print("Usage: python run_bootstrap_from_unimorph.py test_file.txt preds1.txt preds2.txt")
        sys.exit(1)

    test_path, pred1_path, pred2_path = sys.argv[1:]
    test_data = read_unimorph(test_path, testing=False)

    gold_labels = [label for (_, label) in test_data]
    pred1 = read_labels_from_file(pred1_path)
    pred2 = read_labels_from_file(pred2_path)

    assert len(gold_labels) == len(pred1) == len(pred2), "Mismatch in label lengths"

    delta, p_value = paired_bootstrap(gold_labels, pred1, pred2)

    print(f"Δ (accuracy difference): {delta:.4f}")
    print(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("→ Statistically significant difference.")
    else:
        print("→ No significant difference found.")

    # Save to CSV
    csv_path = "bootstrap_results.csv"
    header = ["Test Set", "System A", "System B", "Delta", "P-Value", "Significant"]
    row = [test_path, pred1_path, pred2_path, f"{delta:.4f}", f"{p_value:.4f}", "YES" if p_value < 0.05 else "NO"]

    write_header = not Path(csv_path).exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    print(f"Results written to {csv_path}")

if __name__ == "__main__":
    main()
