"""hmm_tagger.py: A simple HMM-based POS tagger for CoNLL-U files"""
import pyconll
import math
import sys


class HMMTagger:
    """A simple HMM-based POS tagger for CoNLL-U files
    Attributes:
    tag_transitions (dict): Probability of tag following another tag
    tag_emissions (dict): Probability of word having a tag
    tags (set): Set of all possible

    Methods:
    init(self)
        Initialize the tagger with empty transition and emission probabilities
    train(filename)
        Load CoNLL-U file and collect statistics for Viterbi algorithm
    predict(words)
        Use Viterbi algorithm to find most probable tag sequence
    test(filename)
        Return accuracy (correct tags / total tags) on the given test file
    """

    def __init__(self):
        """Initialize the tagger with empty transition and emission probabilities"""
        self.tag_transitions = {}  # Probability of tag following a tag
        self.tag_emissions = {}  # Probability of word having a tag
        self.tags = set()  # Set of all possible tags

    def train(self, filename: str) -> None:
        """Load CoNLL-U file and collect statistics for Viterbi algorithm"""
        training_data = pyconll.load_from_file(filename)

        # Initialize counters
        tag_counts = {}  # Count of each tag
        transition_counts = {}  # Count of tag pairs
        emission_counts = {}  # Count of word-tag pairs

        # Process each sentence
        for sentence in training_data:
            prev_tag = "<s>"  # Padding with start symbol

            for token in sentence:
                # Skip invalid tokens
                if (
                    token.form is None
                    or token.upos is None
                    or token.upos == "PUNCT"
                    or token.form.strip() == ""
                ):
                    continue

                tag = token.upos
                word = token.form.lower()  # Lowercasing

                # Add to set of possible tags
                self.tags.add(tag)

                # Count tag occurrences
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

                # Count transitions (prev_tag -> tag)
                if prev_tag not in transition_counts:
                    transition_counts[prev_tag] = {}
                transition_counts[prev_tag][tag] = (
                    transition_counts[prev_tag].get(tag, 0) + 1
                )

                # Count emissions (tag -> word)
                if tag not in emission_counts:
                    emission_counts[tag] = {}
                emission_counts[tag][word] = (
                    emission_counts[tag].get(word, 0) + 1
                )

                prev_tag = tag

            # Add transition to end symbol for each sentence
            if prev_tag not in transition_counts:
                transition_counts[prev_tag] = {}
            transition_counts[prev_tag]["</s>"] = (
                transition_counts[prev_tag].get("</s>", 0) + 1
            ) # Padding with end symbol

        # Convert counts to probabilities
        self.tag_transitions = {t1: {} for t1 in transition_counts}
        for t1 in transition_counts:
            total = sum(transition_counts[t1].values())
            for t2 in transition_counts[t1]:
                self.tag_transitions[t1][t2] = (
                    transition_counts[t1][t2] / total
                )

        self.tag_emissions = {tag: {} for tag in emission_counts}
        for tag in emission_counts:
            total = sum(emission_counts[tag].values())
            for word in emission_counts[tag]:
                self.tag_emissions[tag][word] = (
                    emission_counts[tag][word] / total
                )

        # Store total counts for use in prediction (if needed)
        self.tag_counts = tag_counts

        print(f"Learned {len(self.tags)} unique tags")
        print(
            f"First few transitions:", list(self.tag_transitions.items())[:2]
        )
        print(
            f"Sample emissions for NOUN:",
            list(self.tag_emissions.get("NOUN", {}).items())[:5],
        )

    def predict(self, words: list[str]) -> list[str]:
        """Use Viterbi algorithm to find most probable sequence of tags
        for the given list of words"""
        n = len(words)
        # Viterbi table: V[t][tag] is the best log-probability up
        # to word position t ending in tag.
        V = [{} for _ in range(n + 1)]
        # Backpointers: B[t][tag] stores the previous tag that gave
        # the best probability to reach tag at position t.
        B = [{} for _ in range(n + 1)]

        # Initialize with the start symbol at position 0.
        V[0]["<s>"] = 0.0  # log probability of 1 is 0

        # Iterate over positions 1 to n
        for t in range(1, n + 1):
            word = words[t - 1].lower()
            for curr_tag in self.tags:
                best_prob = float("-inf")
                best_prev = None
                # Consider every tag at position t-1 that we reached.
                for prev_tag in V[t - 1]:
                    # Get transition probability (with a small fallback for unseen transitions)
                    trans_prob = self.tag_transitions.get(prev_tag, {}).get(
                        curr_tag, 1e-6
                    )
                    # Get emission probability (again, fallback for unknown words)
                    emiss_prob = self.tag_emissions.get(curr_tag, {}).get(
                        word, 1e-6
                    )
                    prob = (
                        V[t - 1][prev_tag]
                        + math.log(trans_prob)
                        + math.log(emiss_prob)
                    )
                    if prob > best_prob:
                        best_prob = prob
                        best_prev = prev_tag
                if best_prev is not None:
                    V[t][curr_tag] = best_prob
                    B[t][curr_tag] = best_prev

        # Termination: transition from last tag to the end symbol '</s>'
        best_final_prob = float("-inf")
        best_final_tag = None
        for tag in V[n]:
            trans_prob = self.tag_transitions.get(tag, {}).get("</s>", 1e-6)
            prob = V[n][tag] + math.log(trans_prob)
            if prob > best_final_prob:
                best_final_prob = prob
                best_final_tag = tag

        # Backtrack to recover the best tag sequence
        best_tags = [None] * n
        current_tag = best_final_tag
        for t in range(n, 0, -1):
            best_tags[t - 1] = current_tag
            current_tag = B[t].get(current_tag, None)
        return best_tags

    def test(self, filename: str) -> float:
        """Return accuracy (correct tags / total tags) on the given test file"""
        test_data = pyconll.load_from_file(filename)
        correct = 0
        total = 0

        for sentence in test_data:
            words = []
            gold_tags = []
            for token in sentence:
                if (
                    token.form is None
                    or token.upos is None
                    or token.upos == "PUNCT"
                    or token.form.strip() == ""
                ):
                    continue
                words.append(token.form)
                gold_tags.append(token.upos)
            if not words:
                continue
            predicted_tags = self.predict(words)
            for pred, gold in zip(predicted_tags, gold_tags):
                if pred == gold:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        print(f"Test accuracy: {accuracy:.2%}")
        return accuracy


def main():
    """Train and evaluate the HMM tagger on the CoNLL-U data"""
    # Allow command-line specification of training and dev files
    if len(sys.argv) > 1:
        train_file = sys.argv[1]
    else:
        train_file = "/home/dsv/robe/lis050/lab1/en_ewt-ud-train.conllu"

    if len(sys.argv) > 2:
        dev_file = sys.argv[2]
    else:
        dev_file = "/home/dsv/robe/lis050/lab1/en_ewt-ud-dev.conllu"

    tagger = HMMTagger()
    print(f"Training on: {train_file}")
    tagger.train(train_file)

    print(f"\nEvaluating on dev set: {dev_file}")
    accuracy = tagger.test(dev_file)
    print(f"\nDev accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
