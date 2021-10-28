from process import ProcessData
from nltk.tokenize import word_tokenize
import string


class UniGram:
    def __init__(self) -> None:
        self.N = 1
        self.unigram_prob = None

        # UNIGRAM ADD ONE SMOOTHING
        self.one_gram_add_one_prob = None

        # UNIGRAM GOOD TURING RESULTS
        self.unigram_good_turing = None
        self.unigram_zero_occurence_prob = None
        self.unigram_good_turing_cstar = None

    def get_unigram_prob(self, sentence: str):
        """Return unigram probability of a given sentences"""
        score = 1
        for word in sentence.split():
            score *= self.unigram_prob[word] if word in self.unigram_prob else 0
        return score

    def get_unigram_add_one_prob(self, sentence: str):
        """Returns unigram probability of a given sentence with add one smoothing"""
        score = 1
        for word in sentence.split():
            if word in self.one_gram_add_one_prob:
                score *= self.one_gram_add_one_prob[word]
            else:
                score *= 0
        return score

    def get_unigram_good_turing_prob(self, sentence: str):
        """Return unigram probability of the given sentence with good turing smoothing"""
        score = 1
        for word in sentence.split():
            if word in self.unigram_good_turing:
                score *= self.unigram_good_turing[word]
            else:
                score *= self.unigram_zero_occurence_prob
        return score

    def train(self, sentences: list):
        unigramCounts = {}
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            words = [word.lower() for word in tokens if word not in string.punctuation]
            if words:
                for word_ in words:
                    if word_ in unigramCounts:
                        unigramCounts[word_] += 1
                    else:
                        unigramCounts[word_] = 1

        return unigramCounts

    def calculate_unigram_prob(self, unigram_counts: dict):
        """Assigns unigram probabilities"""
        total_word_count = sum(unigram_counts.values())

        unigram_prob = {
            unigram: (unigram_counts[unigram]) / total_word_count
            for unigram in unigram_counts
        }

        with open("outputs/unigram/1gram.txt", "w") as file:
            file.write("Unigram" + "\t\t\t" + "Count" + "\t" + "Probability" + "\n")
            for unigram in unigram_counts:
                file.write(
                    str(unigram)
                    + " : "
                    + str(unigram_counts[unigram])
                    + " : "
                    + str(unigram_prob[unigram])
                    + "\n"
                )
        self.unigram_prob = unigram_prob
        return unigram_prob

    def unigram_add_one_smoothing(self, unigram_counts: dict):
        """Assigns unigram probabilities with add one smoothing"""
        total_word_count = sum(unigram_counts.values())

        V = len(unigram_counts)
        one_gram_add_one_prob = {
            unigram: (unigram_counts.get(unigram) + 1) / (total_word_count + V)
            for unigram in unigram_counts
        }
        with open("outputs/unigram/1gram-add-one.txt", "w") as file:
            file.write("Onegram" + "\t\t\t" + "Count" + "\t" + "Probability" + "\n")
            for unigram in unigram_counts:
                file.write(
                    str(unigram)
                    + " : "
                    + str(unigram_counts[unigram])
                    + " : "
                    + str(one_gram_add_one_prob[unigram])
                    + "\n"
                )

        self.one_gram_add_one_prob = one_gram_add_one_prob
        return one_gram_add_one_prob

    def good_turing_smoothing(
        self,
        list_of_word_unit: list,
        word_unit_counts: dict,
        filename_debug: str,
        filename_result: str,
    ):  # sourcery no-metrics
        """Assigns good turing smoothed probabilities"""
        list_of_probabilities = {}
        bucket = {}
        c_star = {}
        p_star = {}
        list_of_counts = {}

        # del word_unit_counts["<s>"]
        # del word_unit_counts["</s>"]

        N = sum(word_unit_counts.values())

        for word_unit in word_unit_counts.items():
            value = word_unit[1]
            if value not in bucket:
                bucket[value] = 1
            else:
                bucket[value] += 1

        bucket_list = sorted(bucket.items(), key=lambda t: t[0])
        # N1 / N
        zero_occurence_prob = bucket_list[0][1] / N
        last_item = bucket_list[len(bucket_list) - 1][0]

        # Set non existing # of words
        for x in range(1, last_item):
            if x not in bucket:
                bucket[x] = 0

        bucket_list = sorted(bucket.items(), key=lambda t: t[0])

        with open(filename_debug, "w") as file:
            file.write("#NumberOfOccurences\t\t\tFrequency\n")
            for c, nc in bucket_list:
                file.write(str(c) + " : " + str(nc) + "\n")
                if nc != 0 and c == last_item or nc == 0:
                    c_star[c] = 0
                    p_star[c] = 0
                else:
                    c_star[c] = (c + 1) * (bucket_list[c + 1][1]) / nc
                    p_star[c] = c_star[c] / N
        for word_unit in list_of_word_unit:
            list_of_probabilities[word_unit] = p_star[word_unit_counts[word_unit]]
            list_of_counts[word_unit] = c_star[word_unit_counts[word_unit]]

        with open(filename_result, "w") as file:
            file.write("Word Unit" + "\t\t\t" + "Count" + "\t" + "Probability" + "\n")

            for unigrams in list_of_word_unit:
                file.write(
                    str(unigrams)
                    + " : "
                    + str(word_unit_counts[unigrams])
                    + " : "
                    + str(list_of_probabilities[unigrams])
                    + "\n"
                )

        self.unigram_good_turing = list_of_probabilities
        self.unigram_zero_occurence_prob = zero_occurence_prob
        self.unigram_good_turing_cstar = list_of_counts

        return list_of_probabilities, zero_occurence_prob, list_of_counts


if __name__ == "__main__":
    x = ProcessData("training_data.txt", 1)
    x.modify_data()
    train_data = x.get_lines()

    ngram = UniGram()

    unigram_counts = ngram.train(train_data)
    unigram_prob = ngram.calculate_unigram_prob(unigram_counts)

    unigram_add_one_prob = ngram.unigram_add_one_smoothing(unigram_counts)

    (
        list_of_probabilities,
        zero_occurence_prob,
        list_of_counts,
    ) = ngram.good_turing_smoothing(
        list(unigram_counts.keys()),
        unigram_counts,
        "outputs/unigram/good_turing_smoothing_unigram.txt",
        "outputs/unigram/good_turing_smoothing_unigram_result.txt",
    )
