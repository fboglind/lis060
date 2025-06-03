"""split_unimorph.py
Splits a UniMorph-formatted file into training and test sets (default 80/20 split)
Usage:
    python split_unimorph.py rus
    python split_unimorph.py bul
""" 
from pathlib import Path
import random
import sys
import random


def split_unimorph_file(lang, split_ratio=0.8, seed=42):
    base_path = Path("data/unimorph")
    infile = base_path / lang
    out_train = base_path / f"{lang}-train"
    out_test = base_path / f"{lang}-test"

    with open(infile, encoding="utf-8") as f:
        lines = f.readlines()

    random.seed(seed)
    random.shuffle(lines)

    split_point = int(len(lines) * split_ratio)
    with open(out_train, "w", encoding="utf-8") as ftrain:
        ftrain.writelines(lines[:split_point])
    with open(out_test, "w", encoding="utf-8") as ftest:
        ftest.writelines(lines[split_point:])

    print(f"Done! {split_point} lines written to {out_train}, {len(lines)-split_point} lines to {out_test}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split_unimorph.py <langcode>")
        sys.exit(1)
    lang = sys.argv[1]
    split_unimorph_file(lang)

