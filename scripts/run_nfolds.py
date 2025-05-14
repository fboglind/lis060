from mynltk.utils.categorized_corpus import CategorizedCorpus
from mynltk.naive_bayes_text_classifier import NaiveBayesTextClassifier
from mynltk.nfolds_cross_validation import CrossValidation

def main():
    # Example usage
    #corpus_path = "data/imdb/test" #[0.8072, 0.8, 0.786, 0.8004, 0.7956, 0.8784, 0.8696, 0.8716, 0.8748, 0.8796], Mean: 0.8363
    corpus_path = "data/sentiment_sentences/imdb-test.tsv" #[0.7, 0.74, 0.76, 0.82, 0.74, 0.8, 0.8, 0.68, 0.78, 0.68], Mean: 0.7500
    cc = CategorizedCorpus(corpus_path)
    data_points = list(cc)  # Convert the generator to a list
    cross_val = CrossValidation()
    cross_val.n_fold(data_points, CrossValidation.evaluate_fold, n=10) # pass a static method as arg
    scores = cross_val.n_fold(
        data_points, 
        CrossValidation.evaluate_fold,  # Static method
        n=10
    )
    
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {sum(scores)/len(scores):.4f}")
    
if __name__ == "__main__":
    main()