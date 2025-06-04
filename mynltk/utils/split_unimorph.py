"""split.py
Resizes datasets to match smallest, then splits into train/test sets.

Usage:
    python split_unimorph.py rus bul
"""
from pathlib import Path
import random
import sys
import argparse


def process_unimorph_datasets(langs, split_ratio=0.8, seed=42):
    """Resize all datasets to smallest size, then split train/test."""
    base_path = Path("data/unimorph")
    
    # Step 1: Read all datasets and find minimum size
    print("Step 1: Reading datasets...")
    datasets = {}
    sizes = {}
    
    for lang in langs:
        filepath = base_path / lang
        with open(filepath, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        datasets[lang] = lines
        sizes[lang] = len(lines)
        print(f"  {lang}: {sizes[lang]} examples")
    
    # Step 2: Find minimum size
    min_size = min(sizes.values())
    min_lang = [lang for lang, size in sizes.items() if size == min_size][0]
    print(f"\nStep 2: Resizing all datasets to match {min_lang} ({min_size} examples)")
    
    # Step 3: Resize larger datasets
    random.seed(seed)
    for lang in langs:
        if sizes[lang] > min_size:
            print(f"  Sampling {min_size} examples from {lang}")
            datasets[lang] = random.sample(datasets[lang], min_size)
        else:
            print(f"  {lang} already at target size")
    
    # Step 4: Split each dataset 80/20
    print(f"\nStep 3: Splitting datasets {int(split_ratio*100)}/{int((1-split_ratio)*100)}")
    
    for lang in langs:
        data = datasets[lang]
        random.seed(seed)  # Reset seed for consistent splits
        random.shuffle(data)
        
        split_point = int(len(data) * split_ratio)
        train_data = data[:split_point]
        test_data = data[split_point:]
        
        # Write files
        train_path = base_path / f"{lang}-train"
        test_path = base_path / f"{lang}-test"
        
        with open(train_path, "w", encoding="utf-8") as f:
            f.writelines(train_data)
        with open(test_path, "w", encoding="utf-8") as f:
            f.writelines(test_data)
        
        print(f"  {lang}: {len(train_data)} train, {len(test_data)} test")
    
    print("\nDone! All datasets now have equal sizes with proportional splits.")


def main():
    parser = argparse.ArgumentParser(description='Resize and split UniMorph datasets')
    parser.add_argument('langs', nargs='+', help='Language codes (e.g., rus bul)')
    parser.add_argument('--split-ratio', type=float, default=0.8, help='Train/test split ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    process_unimorph_datasets(args.langs, args.split_ratio, args.seed)


if __name__ == "__main__":
    main()

# """split_unimorph.py
# Splits UniMorph files into train/test sets and optionally resizes to match smaller dataset.

# Usage:
#     # Just split files (80/20)
#     python split_unimorph.py --split rus bul
    
#     # Split and resize to match smallest
#     python split_unimorph.py --split --resize rus bul
    
#     # Just resize existing train files to match smallest
#     python split_unimorph.py --resize-only rus bul
# """
# from pathlib import Path
# import random
# import sys
# import argparse


# def count_lines(filepath):
#     """Count lines in a file."""
#     with open(filepath, encoding="utf-8") as f:
#         return sum(1 for line in f if line.strip())


# def split_unimorph_file(lang, split_ratio=0.8, seed=42):
#     """Split a UniMorph file into train/test sets."""
#     base_path = Path("data/unimorph")
#     infile = base_path / lang
#     out_train = base_path / f"{lang}-train"
#     out_test = base_path / f"{lang}-test"

#     with open(infile, encoding="utf-8") as f:
#         lines = [line for line in f if line.strip()]  # Skip empty lines

#     random.seed(seed)
#     random.shuffle(lines)

#     split_point = int(len(lines) * split_ratio)
    
#     with open(out_train, "w", encoding="utf-8") as ftrain:
#         ftrain.writelines(lines[:split_point])
#     with open(out_test, "w", encoding="utf-8") as ftest:
#         ftest.writelines(lines[split_point:])

#     print(f"{lang}: Split {len(lines)} lines into {split_point} train, {len(lines)-split_point} test")
#     return split_point, len(lines) - split_point


# def resize_dataset(filepath, target_size, seed=42):
#     """Resize a dataset file to target size by sampling."""
#     with open(filepath, encoding="utf-8") as f:
#         lines = [line for line in f if line.strip()]
    
#     current_size = len(lines)
    
#     if current_size <= target_size:
#         print(f"  {filepath.name}: Already smaller ({current_size}) than target ({target_size})")
#         return current_size
    
#     random.seed(seed)
#     sampled_lines = random.sample(lines, target_size)
    
#     # Create backup of original
#     backup_path = filepath.with_suffix(filepath.suffix + '.original')
#     if not backup_path.exists():
#         filepath.rename(backup_path)
#         print(f"  Backed up original to {backup_path.name}")
    
#     # Write sampled data
#     with open(filepath, "w", encoding="utf-8") as f:
#         f.writelines(sampled_lines)
    
#     print(f"  {filepath.name}: Resized from {current_size} to {target_size}")
#     return target_size


# def get_dataset_sizes(langs, data_type='train'):
#     """Get sizes of datasets for given languages."""
#     base_path = Path("data/unimorph")
#     sizes = {}
    
#     for lang in langs:
#         filepath = base_path / f"{lang}-{data_type}"
#         if filepath.exists():
#             sizes[lang] = count_lines(filepath)
#         else:
#             print(f"Warning: {filepath} not found")
#             sizes[lang] = 0
    
#     return sizes


# def resize_to_match_smallest(langs, data_type='train', seed=42):
#     """Resize all datasets to match the smallest one."""
#     base_path = Path("data/unimorph")
    
#     # Get current sizes
#     sizes = get_dataset_sizes(langs, data_type)
    
#     if not sizes:
#         print("No datasets found!")
#         return
    
#     # Find minimum size
#     min_size = min(sizes.values())
#     min_lang = [lang for lang, size in sizes.items() if size == min_size][0]
    
#     print(f"\nResizing {data_type} sets to match {min_lang} ({min_size} examples):")
    
#     # Resize each dataset
#     for lang in langs:
#         filepath = base_path / f"{lang}-{data_type}"
#         if filepath.exists():
#             resize_dataset(filepath, min_size, seed)


# def main():
#     parser = argparse.ArgumentParser(description='Split and resize UniMorph datasets')
#     parser.add_argument('langs', nargs='+', help='Language codes (e.g., rus bul)')
#     parser.add_argument('--split', action='store_true', help='Split original files into train/test')
#     parser.add_argument('--resize', action='store_true', help='Resize to match smallest dataset')
#     parser.add_argument('--resize-only', action='store_true', help='Only resize existing train/test files')
#     parser.add_argument('--split-ratio', type=float, default=0.8, help='Train/test split ratio (default: 0.8)')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
#     parser.add_argument('--data-type', choices=['train', 'test', 'both'], default='train',
#                        help='Which dataset to resize (default: train)')
    
#     args = parser.parse_args()
    
#     if not (args.split or args.resize or args.resize_only):
#         print("Error: Specify at least one of --split, --resize, or --resize-only")
#         parser.print_help()
#         sys.exit(1)
    
#     print(f"Processing languages: {', '.join(args.langs)}")
#     print(f"Random seed: {args.seed}")
    
#     # Step 1: Split files if requested
#     if args.split and not args.resize_only:
#         print("\nStep 1: Splitting files...")
#         for lang in args.langs:
#             split_unimorph_file(lang, args.split_ratio, args.seed)
    
#     # Step 2: Resize if requested
#     if args.resize or args.resize_only:
#         print("\nStep 2: Resizing datasets...")
        
#         # Show current sizes
#         print("\nCurrent dataset sizes:")
#         for data_type in ['train', 'test']:
#             sizes = get_dataset_sizes(args.langs, data_type)
#             for lang, size in sizes.items():
#                 print(f"  {lang}-{data_type}: {size}")
        
#         # Resize based on data_type argument
#         if args.data_type in ['train', 'both']:
#             resize_to_match_smallest(args.langs, 'train', args.seed)
        
#         if args.data_type in ['test', 'both']:
#             resize_to_match_smallest(args.langs, 'test', args.seed)
    
#     print("\nDone!")
    
#     # Show final sizes
#     print("\nFinal dataset sizes:")
#     for data_type in ['train', 'test']:
#         sizes = get_dataset_sizes(args.langs, data_type)
#         for lang, size in sizes.items():
#             print(f"  {lang}-{data_type}: {size}")


# if __name__ == "__main__":
#     main()

# """split_unimorph.py
# Splits a UniMorph-formatted file into training and test sets (default 80/20 split)
# Usage:
#     python split_unimorph.py rus
#     python split_unimorph.py bul
# """ 
# from pathlib import Path
# import random
# import sys
# import random


# def split_unimorph_file(lang, split_ratio=0.8, seed=42):
#     base_path = Path("data/unimorph")
#     infile = base_path / lang
#     out_train = base_path / f"{lang}-train"
#     out_test = base_path / f"{lang}-test"

#     with open(infile, encoding="utf-8") as f:
#         lines = f.readlines()

#     random.seed(seed)
#     random.shuffle(lines)

#     split_point = int(len(lines) * split_ratio)
#     with open(out_train, "w", encoding="utf-8") as ftrain:
#         ftrain.writelines(lines[:split_point])
#     with open(out_test, "w", encoding="utf-8") as ftest:
#         ftest.writelines(lines[split_point:])

#     print(f"Done! {split_point} lines written to {out_train}, {len(lines)-split_point} lines to {out_test}")

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python split_unimorph.py <langcode>")
#         sys.exit(1)
#     lang = sys.argv[1]
#     split_unimorph_file(lang)

