from categorized_corpus import CategorizedCorpus

cc = CategorizedCorpus("data/imdb/train")
for text, category in cc:
    print(f"{text[:40]}... {category}")
