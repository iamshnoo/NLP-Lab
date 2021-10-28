from process import ProcessData
from bigram import BiGram
from nltk.tokenize import word_tokenize
import string


class TriGram:
    def __init__(self) -> None:
        self.N = 3
        self.trigram_prob = None

        # TRIGRAM ADD ONE SMOOTHING
        self.trigram_add_one_prob = None

        # TRIGRAM GOOD TURING RESULTS
        self.trigram_good_turing = None
        self.trigram_zero_occurence_prob = None
        self.trigram_good_turing_cstar = None

    def get_trigram_prob(self, sentence: str):
        """Returns trigram probability of a given sentence"""
        score = 1
        words = sentence.split()
        for i in range(len(words) - 2):
            unit = (words[i], words[i + 1], words[i + 2])
            score *= self.trigram_prob[unit] if unit in self.trigram_prob else 0
        return score

    def get_trigram_add_one_prob(self, sentence: str):
        """Returns trigram probability of a given sentence with add one smoothing"""
        score = 1
        words = sentence.split()
        for i in range(len(words) - 2):
            unit = tuple(words[i : i + 3])
            if unit in self.trigram_add_one_prob:
                score *= self.trigram_add_one_prob[unit]
            else:
                score *= 0
        return score

    def get_trigram_good_turing_prob(self, sentence: str):
        """Returns trigram probability of a given sentence with good turing smoothing"""
        score = 1
        words = sentence.split()
        for i in range(len(words) - 2):
            unit = (words[i], words[i + 1], words[i + 2])
            if unit in self.trigram_good_turing:
                score *= self.trigram_good_turing[unit]
            else:
                score *= self.trigram_zero_occurence_prob
        return score

    def train(self, sentences: list):

        listOfTrigrams = []
        trigramCounts = {}

        for sentence in sentences:
            tokens = word_tokenize(sentence)
            words = [word.lower() for word in tokens if word not in string.punctuation]
            if words:
                for i in range(len(words)):
                    if i < len(words) - 2:
                        unit = (words[i], words[i + 1], words[i + 2])
                        listOfTrigrams.append(unit)
                        if unit in trigramCounts:
                            trigramCounts[unit] += 1
                        else:
                            trigramCounts[unit] = 1

        return listOfTrigrams, trigramCounts

    def calculate_trigram_prob(
        self, list_of_trigrams: list, bigram_counts: dict, trigram_counts: dict
    ):
        """Assigns trigram probabilities"""
        trigram_prob = {
            trigram: (trigram_counts[trigram])
            / (bigram_counts[(trigram[0], trigram[1])])
            for trigram in list_of_trigrams
        }
        with open("outputs/trigram/3gram.txt", "w") as file:
            file.write("Trigram" + "\t\t\t" + "Count" + "\t" + "Probability" + "\n")
            for bigrams in list_of_trigrams:
                file.write(
                    str(bigrams)
                    + " : "
                    + str(trigram_counts[bigrams])
                    + " : "
                    + str(trigram_prob[bigrams])
                    + "\n"
                )
        self.trigram_prob = trigram_prob
        return trigram_prob

    def trigram_add_one_smoothing(self, trigram_counts: dict):
        """Assigns trigram probabilities with add one smoothing"""
        total_word_count = sum(trigram_counts.values())

        V = len(trigram_counts)
        tri_gram_add_one_prob = {
            trigram: (trigram_counts[trigram] + 1) / (total_word_count + V)
            for trigram in trigram_counts
        }
        with open("outputs/trigram/3gram-add-one.txt", "w") as file:
            file.write("Trigram" + "\t\t\t" + "Count" + "\t" + "Probability" + "\n")
            for trigram in trigram_counts:
                file.write(
                    str(trigram)
                    + " : "
                    + str(trigram_counts[trigram])
                    + " : "
                    + str(tri_gram_add_one_prob[trigram])
                    + "\n"
                )

        self.trigram_add_one_prob = tri_gram_add_one_prob
        return tri_gram_add_one_prob

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

            for trigrams in list_of_word_unit:
                file.write(
                    str(trigrams)
                    + " : "
                    + str(word_unit_counts[trigrams])
                    + " : "
                    + str(list_of_probabilities[trigrams])
                    + "\n"
                )

        self.trigram_good_turing = list_of_probabilities
        self.trigram_zero_occurence_prob = zero_occurence_prob
        self.trigram_good_turing_cstar = list_of_counts

        return list_of_probabilities, zero_occurence_prob, list_of_counts


if __name__ == "__main__":
    x = ProcessData("training_data.txt", 3)
    x.modify_data()
    train_data = x.get_lines()
    bigram = BiGram()
    _, bigram_counts = bigram.train(train_data)

    ngram = TriGram()
    list_of_trigrams, trigram_counts = ngram.train(train_data)

    trigram_prob = ngram.calculate_trigram_prob(
        list_of_trigrams, bigram_counts, trigram_counts
    )

    tri_gram_add_one_prob = ngram.trigram_add_one_smoothing(trigram_counts)

    (
        list_of_probabilities,
        zero_occurence_prob,
        list_of_counts,
    ) = ngram.good_turing_smoothing(
        list_of_trigrams,
        trigram_counts,
        "outputs/trigram/good_turing_smoothing_trigram.txt",
        "outputs/trigram/good_turing_smoothing_trigram_result.txt",
    )
