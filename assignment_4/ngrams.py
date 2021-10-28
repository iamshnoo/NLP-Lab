from typing import Optional
from nltk.tokenize import word_tokenize
import string
import math
from utils import testing_data, training_data, process_sentence
from contextlib import redirect_stdout


class NGram(object):
    def __init__(self, n: int) -> None:
        self.N = n

        self.ngram_prob = None

        # NGRAM ADD ONE SMOOTHING
        self.ngram_add_one_prob = None

        # NGRAM GOOD TURING RESULTS
        self.ngram_good_turing = None
        self.ngram_zero_occurence_prob = None
        self.ngram_good_turing_cstar = None

    def perplexity(self, sentence: str, smoothing: Optional[str] = None):
        """Returns ngram perplexity of a given sentence for a calculated score
        (unsmoothed / smoothed)."""
        counter = 0
        temp = 0
        sentence = process_sentence(sentence)
        for j in range(len(sentence) - self.N + 1):
            unit = tuple(sentence[j : j + self.N])
            if smoothing:
                if smoothing == "add_one":
                    prob = (
                        self.ngram_add_one_prob[unit]
                        if unit in self.ngram_add_one_prob
                        else 0
                    )
                if smoothing == "good_turing":
                    prob = (
                        self.ngram_good_turing[unit]
                        if unit in self.ngram_add_one_prob
                        else self.ngram_zero_occurence_prob
                    )
            else:
                prob = self.ngram_prob[unit] if unit in self.ngram_prob else 0
            # temp is sum of log probabilities of ngrams in the sentence
            temp += math.log(prob, 2) if prob else 0
            # counter counts the number of ngrams (not number of words)
            counter += 1
        # entropy = prob of each token / number of tokens
        entropy = (-1 / counter) * temp

        # perplexity
        return math.pow(2, entropy)

    def fit(self, sentences: list):
        """Fits the model to training sentences for smoothed and unsmoothed
        versions simultaneously."""
        list_of_ngrams, ngram_counts = self.train(sentences)
        ngram_prob = self.calculate_ngram_prob(ngram_counts)
        n_gram_add_one_prob = self.ngram_add_one_smoothing(ngram_counts)
        (
            list_of_probabilities,
            zero_occurence_prob,
            list_of_counts,
        ) = self.good_turing_smoothing(
            list_of_ngrams,
            ngram_counts,
            "outputs/ngram/good_turing_smoothing_ngram.txt",
            "outputs/ngram/good_turing_smoothing_ngram_result.txt",
        )

    def get_ngram_prob(self, sentence: str):
        """Return ngram probability of a given sentence"""
        score = 1
        words = sentence.split()
        for i in range(len(words) - self.N + 1):
            unit = tuple(words[i : i + self.N])
            score *= self.ngram_prob[unit] if unit in self.ngram_prob else 0

        return score

    def get_ngram_add_one_prob(self, sentence: str):
        """Returns ngram probability of a given sentence with add one smoothing"""
        score = 1
        words = sentence.split()
        for i in range(len(words) - self.N + 1):
            unit = tuple(words[i : i + self.N])
            score *= (
                self.ngram_add_one_prob[unit] if unit in self.ngram_add_one_prob else 0
            )

        return score

    def get_ngram_good_turing_prob(self, sentence: str):
        """Return ngram probability of the given sentence with good turing smoothing"""
        score = 1
        words = sentence.split()
        for i in range(len(words) - self.N + 1):
            unit = tuple(words[i : i + self.N])
            score *= (
                self.ngram_good_turing[unit]
                if unit in self.ngram_good_turing
                else self.ngram_zero_occurence_prob
            )
        return score

    def train(self, sentences: list):
        listOfNgrams = []
        ngramCounts = {}

        self.n_minus_one_gramCounts = {}

        for sentence in sentences:
            tokens = word_tokenize(sentence)
            words = [word.lower() for word in tokens if word not in string.punctuation]
            if words:
                for i in range(len(words) - self.N + 1):
                    unit = tuple(words[i : i + self.N])
                    if self.N == 2:
                        n_minus_one_th_unit = words[i]
                    if self.N > 2:
                        n_minus_one_th_unit = tuple(words[i : i + self.N - 1])
                    listOfNgrams.append(unit)
                    if unit in ngramCounts:
                        ngramCounts[unit] += 1
                    else:
                        ngramCounts[unit] = 1

                    if self.N != 1:
                        if n_minus_one_th_unit in self.n_minus_one_gramCounts:
                            self.n_minus_one_gramCounts[n_minus_one_th_unit] += 1
                        else:
                            self.n_minus_one_gramCounts[n_minus_one_th_unit] = 1

        self.list_of_ngrams = listOfNgrams
        return listOfNgrams, ngramCounts

    def calculate_ngram_prob(self, ngram_counts: dict):
        """Assigns ngram probabilities"""

        ngram_prob = {}
        total_word_count = sum(ngram_counts.values())

        if self.N == 1:
            for ngram in ngram_counts:
                ngram_prob[ngram] = (ngram_counts[ngram]) / total_word_count

        elif self.N == 2:
            for ngram in self.list_of_ngrams:
                ngram_prob[ngram] = (ngram_counts[ngram]) / (
                    self.n_minus_one_gramCounts[ngram[0]]
                )

        elif self.N == 3:
            for ngram in self.list_of_ngrams:
                ngram_prob[ngram] = (ngram_counts[ngram]) / (
                    self.n_minus_one_gramCounts[(ngram[0], ngram[1])]
                )

        else:
            for ngram in self.list_of_ngrams:
                unit = ngram[0 : N - 1]
                ngram_prob[ngram] = (ngram_counts[ngram]) / (
                    self.n_minus_one_gramCounts[unit]
                )

        with open("outputs/ngram/ngram.txt", "w") as file:
            file.write("Ngram" + "\t\t\t" + "Count" + "\t" + "Probability" + "\n")
            for ngrams in self.list_of_ngrams:
                file.write(
                    str(ngrams)
                    + " : "
                    + str(ngram_counts[ngrams])
                    + " : "
                    + str(ngram_prob[ngrams])
                    + "\n"
                )

        self.ngram_prob = ngram_prob
        return ngram_prob

    def ngram_add_one_smoothing(self, ngram_counts: dict):
        """Assigns ngram probabilities with add one smoothing."""

        total_word_count = sum(ngram_counts.values())

        V = len(ngram_counts)
        n_gram_add_one_prob = {
            ngram: (ngram_counts[ngram] + 1) / (total_word_count + V)
            for ngram in ngram_counts
        }
        with open("outputs/ngram/ngram-add-one.txt", "w") as file:
            file.write("Ngram" + "\t\t\t" + "Count" + "\t" + "Probability" + "\n")
            for ngram in ngram_counts:
                file.write(
                    str(ngram)
                    + " : "
                    + str(ngram_counts[ngram])
                    + " : "
                    + str(n_gram_add_one_prob[ngram])
                    + "\n"
                )

        self.ngram_add_one_prob = n_gram_add_one_prob
        return n_gram_add_one_prob

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

            for ngrams in list_of_word_unit:
                file.write(
                    str(ngrams)
                    + " : "
                    + str(word_unit_counts[ngrams])
                    + " : "
                    + str(list_of_probabilities[ngrams])
                    + "\n"
                )

        self.ngram_good_turing = list_of_probabilities
        self.ngram_zero_occurence_prob = zero_occurence_prob
        self.ngram_good_turing_cstar = list_of_counts

        return list_of_probabilities, zero_occurence_prob, list_of_counts


if __name__ == "__main__":
    N = 2
    train_data = training_data(N)
    test_data = testing_data(N)

    ngram = NGram(N)
    ngram.fit(train_data)

    with open("outputs/ngram/ngram_perplexity.txt", "w") as data_file:
        with redirect_stdout(data_file):
            for i, sentence in enumerate(test_data):
                perplexity = ngram.perplexity(sentence, "add_one")
                print("{:>5}{:>30}".format(i, round(perplexity, 2)))
