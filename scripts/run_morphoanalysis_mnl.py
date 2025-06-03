"""run_morphoanalysis.py
A script for testing a morphological analyzer
    Functions: 
        read_unimorph: Read UniMorph data and prepare for analysis
        compute_accuracy: Computes accuracy by comparing test and gold sets
        main(): Main function for execution
"""
import time
#from mynltk import MorphologicalAnalyzer
# from mynltk import MultinomialLogisticRegression
from mynltk.morphological_analyzer import MorphologicalAnalyzer
#from mynltk.multinomial_regression import MultinomialLogisticRegression
from mynltk.utils.read_unimorph import read_unimorph

def compute_accuracy(gold_data, predicted_tags):
    """Computes accuracy by comparing test and gold sets"""
    correct = sum(
        1
        for ((_, _), gold), pred in zip(gold_data, predicted_tags)
        if gold == pred
    )
    return correct / len(gold_data)


DATA_DIR = "data"


def main():
    """Main function for execution"""

    start = time.time()

    # Load data
    print("Loading data..")
    train_data = read_unimorph(
        "data/unimorph/bul-train"
    )  # Returns [(base, inflected, tags)]
    test_data = read_unimorph(
        "data/unimorph/bul-test", testing=True
    )  # Returns [inflected_words]
    gold_data = read_unimorph("data/unimorph/bul-test", testing=False)

    # Check
    # print(test_data[4])
    # print(test_data[6])
    # print(test_data[7])
    # print(gold_data[4])

    #check
    print("Number of unique labels:", len(set(label for _, label in train_data)))
    # Train and analyze

    analyzer = MorphologicalAnalyzer(training_data=train_data, n_iters=3)
    print(f"Analysis started.. analyzing for {analyzer.n_iters} epochs")
    # Predict
    predictions = analyzer.predict(test_data)

    stop = time.time()
    elapsed_time = stop - start
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time passed: {int(minutes)} min, {seconds:.2f} seconds\n")
    # Show top features
    analyzer.show_top_features()

    accuracy = compute_accuracy(gold_data, predictions)

    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
