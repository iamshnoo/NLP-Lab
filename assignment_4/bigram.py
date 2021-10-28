from process import ProcessData
from unigram import UniGram
from nltk.tokenize import word_tokenize
import string


class BiGram:
    def __init__(self) -> None:
        self.N = 2
        self.bigram_prob = None

        # BIGRAM ADD ONE SMOOTHING
        self.bigram_add_one_prob = None

        # BIGRAM GOOD TURING RESULTS
        self.bigram_good_turing = None
        self.bigram_zero_occurence_prob = None
        self.bigram_good_turing_cstar = None

    def get_bigram_prob(self, sentence: str):
        """Return bigram probability of a given sentence"""
        score = 1
        words = sentence.split()
        for i in range(len(words) - 1):
            unit = (words[i], words[i + 1])
            score *= self.bigram_prob[unit] if unit in self.bigram_prob else 0
        return score

    def get_bigram_add_one_prob(self, sentence: str):
        """Returns bigram probability of a given sentence with add one smoothing"""
        score = 1
        words = sentence.split()
        for i in range(len(words) - 1):
            unit = tuple(words[i : i + 2])
            if unit in self.bigram_add_one_prob:
                score *= self.bigram_add_one_prob[unit]
            else:
                score *= 0
        return score

    def get_bigram_good_turing_prob(self, sentence: str):
        """Return bigram probability of a given sentence with good turing smoothing"""
        score = 1
        words = sentence.split()
        for i in range(len(words) - 1):
            unit = (words[i], words[i + 1])
            if unit in self.bigram_good_turing:
                score *= self.bigram_good_turing[unit]
            else:
                score *= self.bigram_zero_occurence_prob
        return score

    def train(self, sentences: list):
        listOfBigrams = []
        bigramCounts = {}

        for sentence in sentences:
            tokens = word_tokenize(sentence)
            words = [word.lower() for word in tokens if word not in string.punctuation]
            if words:
                for i in range(len(words)):
                    if i < len(words) - 1:
                        unit = (words[i], words[i + 1])
                        listOfBigrams.append(unit)
                        if unit in bigramCounts:
                            bigramCounts[unit] += 1
                        else:
                            bigramCounts[unit] = 1

        return listOfBigrams, bigramCounts

    def calculate_bigram_prob(
        self, listOfBigrams: list, unigramCounts: dict, bigramCounts: dict
    ):
        """Assigns bigram probabilities"""
        bigram_prob = {
            bigram: (bigramCounts[bigram]) / (unigramCounts[bigram[0]])
            for bigram in listOfBigrams
        }

        with open("outputs/bigram/2gram.txt", "w") as file:
            file.write("Bigram" + "\t\t\t" + "Count" + "\t" + "Probability" + "\n")
            for bigrams in listOfBigrams:
                file.write(
                    str(bigrams)
                    + " : "
                    + str(bigramCounts[bigrams])
                    + " : "
                    + str(bigram_prob[bigrams])
                    + "\n"
                )
        self.bigram_prob = bigram_prob
        return bigram_prob

    def bigram_add_one_smoothing(self, bigram_counts: dict):
        """Assigns bigram probabilities with add one smoothing"""
        total_word_count = sum(bigram_counts.values())

        V = len(bigram_counts)
        bi_gram_add_one_prob = {
            bigram: (bigram_counts[bigram] + 1) / (total_word_count + V)
            for bigram in bigram_counts
        }
        with open("outputs/bigram/2gram-add-one.txt", "w") as file:
            file.write("Bigram" + "\t\t\t" + "Count" + "\t" + "Probability" + "\n")
            for bigram in bigram_counts:
                file.write(
                    str(bigram)
                    + " : "
                    + str(bigram_counts[bigram])
                    + " : "
                    + str(bi_gram_add_one_prob[bigram])
                    + "\n"
                )

        self.bigram_add_one_prob = bi_gram_add_one_prob
        return bi_gram_add_one_prob

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

            for bigrams in list_of_word_unit:
                file.write(
                    str(bigrams)
                    + " : "
                    + str(word_unit_counts[bigrams])
                    + " : "
                    + str(list_of_probabilities[bigrams])
                    + "\n"
                )

        self.bigram_good_turing = list_of_probabilities
        self.bigram_zero_occurence_prob = zero_occurence_prob
        self.bigram_good_turing_cstar = list_of_counts

        return list_of_probabilities, zero_occurence_prob, list_of_counts


if __name__ == "__main__":
    x = ProcessData("training_data.txt", 2)
    x.modify_data()
    train_data = x.get_lines()

    unigram = UniGram()
    unigram_counts = unigram.train(train_data)

    ngram = BiGram()
    list_of_bigrams, bigram_counts = ngram.train(train_data)

    bigram_prob = ngram.calculate_bigram_prob(
        list_of_bigrams, unigram_counts, bigram_counts
    )

    bi_gram_add_one_prob = ngram.bigram_add_one_smoothing(bigram_counts)

    (
        list_of_probabilities,
        zero_occurence_prob,
        list_of_counts,
    ) = ngram.good_turing_smoothing(
        list_of_bigrams,
        bigram_counts,
        "outputs/bigram/good_turing_smoothing_bigram.txt",
        "outputs/bigram/good_turing_smoothing_bigram_result.txt",
    )
