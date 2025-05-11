from categorized_corpus import CategorizedCorpus

# cc = CategorizedCorpus("data/imdb/train")
# for text, category in cc:
#     print(f"{text[:40]}... {category}")

cc_test = CategorizedCorpus("../data/imdb/test")
#cc_test = CategorizedCorpus("../data/sentiment_sentences/imdb-test.tsv")
for text, category in cc_test:
    print(f"{text[:40]}... {category}")