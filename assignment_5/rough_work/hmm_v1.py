import pprint
import random
import time

import numpy as np
import pandas as pd
from nltk import word_tokenize

from dataset import *

DEMO = True


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class HMM:
    def __init__(self, corpus, seed=42):
        random.seed(seed)
        self.data = Dataset(corpus)
        # create list of train and test tagged words
        self.train_tagged_words = [
            tup for sent in self.data.train_sentences for tup in sent
        ]
        self.test_tagged_words = [
            tup.token for sent in self.data.test_sentences for tup in sent
        ]

        # training vocab and tagset
        self.training_tagset = sorted({word.POS for word in self.train_tagged_words})
        self.training_vocab = sorted({word.token for word in self.train_tagged_words})

        self.token_count = {}

    def calculate_tag_occurrence_probability(self, print_output=True):
        self.tag_prob = []
        total_tag = len([word.POS for word in self.train_tagged_words])
        for t in self.training_tagset:
            each_tag = [word.POS for word in self.train_tagged_words if word.POS == t]
            self.tag_prob.append((t, len(each_tag) / total_tag))

        self.tag_prob = dict(self.tag_prob)
        if print_output:
            print("Tag occurrence probabilities : ")
            print("-" * 100)
            pprint.pprint(self.tag_prob)
            print("-" * 100)
        return self.tag_prob

    def calculate_pi(self, print_output=True):
        # The initial tag probabilities π(ti) for 1<i< n,
        # where π(ti) is the probability that a
        # sentence begins with tag ti.
        sentence_start_words = []
        for sentence in self.data.train_sentences:
            for i, word in enumerate(sentence):
                if i == 0:
                    sentence_start_words.append(word)

        self.pi = []
        total_tag = len([word.POS for word in self.train_tagged_words])
        for t in self.training_tagset:
            each_tag = [word.POS for word in sentence_start_words if word.POS == t]
            self.pi.append((t, len(each_tag) / total_tag))

        self.pi = dict(self.pi)
        if print_output:
            print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
            print(bcolors.OKGREEN + bcolors.BOLD + "Pi : " + bcolors.ENDC)
            print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
            pprint.pprint(self.pi)
            print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)

        return self.pi

    @staticmethod
    def P_tj_given_ti(tj, ti, train_bag):
        """Utility method to find probability of tag tj given tag ti."""
        tags = [word.POS for word in train_bag]
        ti_tags = [tag for tag in tags if tag == ti]
        count_of_ti = len(ti_tags)

        tj_given_ti = [
            tags[index + 1]
            for index in range(len(tags) - 1)
            if tags[index] == ti and tags[index + 1] == tj
        ]

        count_tj_given_ti = len(tj_given_ti)
        return count_tj_given_ti / count_of_ti

    def calculate_transition_matrix(self, print_output=True):
        # The transition probabilities a(ti →tj)
        # for 1<(i, j)<n where a(ti →tj) is the
        # probability that tag tj occurs after tag ti.

        # creating transition matrix
        # no. of cols = no. of rows = no. of hidden states = no. of tags
        self.transition_matrix = np.zeros(
            (len(self.training_tagset), len(self.training_tagset)), dtype="float32"
        )

        # each column is tj, each row is ti
        # M(i, j) represents P(tj given ti)
        for i, ti in enumerate(list(self.training_tagset)):
            for j, tj in enumerate(list(self.training_tagset)):
                self.transition_matrix[i, j] = self.P_tj_given_ti(
                    tj, ti, self.train_tagged_words
                )

        # convert the matrix to a df
        self.transition_df = pd.DataFrame(
            self.transition_matrix,
            columns=list(self.training_tagset),
            index=list(self.training_tagset),
        )

        if print_output:
            print(
                bcolors.OKGREEN + bcolors.BOLD + "Transition matrix : " + bcolors.ENDC
            )
            print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
            print(self.transition_df)
            print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)

        return self.transition_df, self.transition_matrix

    @staticmethod
    def P_word_given_tag(word, tag, train_bag):
        taglist = [item for item in train_bag if item.POS == tag]
        tag_count = len(taglist)
        w_in_tag = [item.token for item in taglist if item.token == word]
        word_count_given_tag = len(w_in_tag)
        return word_count_given_tag / tag_count, tag_count

    def calculate_emission_matrix(self, print_output=True):
        # The emission probabilities b(ti →wj)
        # for 1<i< n and 1<j< n where b(ti →wj) is the
        # probability that token wj is generated given tag ti.

        # creating emission matrix
        # no. of rows = no. of hidden states = no. of tags
        # no. of cols = no. of visible states = no. of words in vocabulary
        self.emission_matrix = np.zeros(
            (len(self.training_tagset), len(self.training_vocab)), dtype="float32"
        )

        # each column is wj, each row is ti
        # M(i, j) represents P(wj given ti)
        for i, ti in enumerate(list(self.training_tagset)):
            for j, wj in enumerate(list(self.training_vocab)):
                self.emission_matrix[i, j], x = self.P_word_given_tag(
                    wj, ti, self.train_tagged_words
                )
                self.token_count[ti] = x

        # convert the matrix to a df
        self.emission_df = pd.DataFrame(
            self.emission_matrix,
            columns=list(self.training_vocab),
            index=list(self.training_tagset),
        )

        if print_output:
            print(bcolors.OKGREEN + bcolors.BOLD + "Emission matrix : " + bcolors.ENDC)
            print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
            print(self.emission_df)
            print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)

        return self.emission_df, self.emission_matrix

    def fit(self, printing=False):
        self.calculate_pi(print_output=printing)
        self.calculate_transition_matrix(print_output=printing)
        self.calculate_emission_matrix(print_output=printing)

    def get_smoothened_emission_probability(self, emission, tag, token):
        total_tokens_from_tag = self.token_count[tag]
        emission_count = (
            emission.loc[tag, token] if token in emission.columns else 0
        ) * total_tokens_from_tag
        vocab_size = len(emission.index)

        return (emission_count + 1) / (total_tokens_from_tag + vocab_size)

    # Vanilla Viterbi Algorithm
    def most_probable_tags(self, words, print_alpha=True, print_output=True):
        # sourcery no-metrics

        # initialize alpha
        alpha = np.zeros((len(self.training_tagset), len(words)), dtype="float32")
        # alpha_df = pd.DataFrame(
        #     alpha,
        #     columns=list(words),
        #     index=list(self.training_tagset),
        # )
        alpha_df = pd.DataFrame(
            alpha,
            columns=[str(i) for i in range(len(words))],
            index=list(self.training_tagset),
        )
        # print(alpha_df)

        # forward pass
        for i, word in enumerate(words):
            for state in self.training_tagset:

                # for unknown words, P_word_given_tag = 0
                emission_prob = self.get_smoothened_emission_probability(
                    self.emission_df, state, words[i]
                )

                if i == 0:
                    alpha_df.loc[state, words[i]] = self.pi[state] * emission_prob
                else:
                    x = alpha_df.loc[self.training_tagset[0], str(i - 1)]
                    # x = alpha_df.loc[self.training_tagset[0], words[i - 1]]
                    # if not isinstance(x, np.float32):
                    #     x = x.drop_duplicates()[0]

                    max_transition_prob = (
                        x * self.transition_df.loc[self.training_tagset[0], state]
                    )
                    previous_state_selected = self.training_tagset[0]

                    for previous_state in self.training_tagset[1:]:
                        y = alpha_df.loc[previous_state, str(i - 1)]
                        # y = alpha_df.loc[previous_state, words[i - 1]]
                        # if not isinstance(y, np.float32):
                        #     y = y.drop_duplicates()[0]

                        transition_prob = (
                            y * self.transition_df.loc[previous_state, state]
                        )
                        if transition_prob > max_transition_prob:
                            max_transition_prob = transition_prob
                            previous_state_selected = previous_state

                    max_prob = max_transition_prob * emission_prob
                    alpha_df.loc[state, words[i]] = max_prob

        # backward pass
        max_prob = alpha_df.iloc[:, -1:].max().values[0]
        max_states = (alpha_df.idxmax()).to_frame()
        states = [row[0] for _, row in max_states.iterrows()]

        if print_output:
            print("Given sentence : ", words)
            print("Predicted tag sequence : ", states, " with probability ", max_prob)
            print("-" * 100)

        if print_alpha:
            print("Alpha matrix : ")
            print("-" * 100)
            print(alpha_df)
            print("-" * 100)

        return [Word(token, POS) for token, POS in zip(words, states)]

    def test(self, test_sentence=None):
        # Checking model performance on a single test sentence
        test = "But my future plans ran east." if not test_sentence else test_sentence
        test_tagged_words = word_tokenize(test)
        tagged_seq = self.most_probable_tags(test_tagged_words)
        print(test)
        print(tagged_seq)

    def test_performance(self):
        # Checking model performance on test data
        total_correct = 0
        total_tagged_seq = 0
        for i, sentence in enumerate(self.data.test_sentences):
            print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
            print(
                bcolors.OKGREEN + bcolors.BOLD + "Test Sentence ",
                i + 1,
                " : " + bcolors.ENDC,
            )
            print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
            test_run_base = [word for word in sentence]
            test_tagged_words = [word.token for word in sentence]

            start = time.time()
            tagged_seq = self.most_probable_tags(test_tagged_words, print_alpha=False)
            end = time.time()
            difference = end - start

            print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
            correct = [i for i, j in zip(tagged_seq, test_run_base) if i == j]
            accuracy = len(correct) / len(tagged_seq)
            total_correct += len(correct)
            total_tagged_seq += len(tagged_seq)
            incorrectly_tagged_words = [
                (i, j) for i, j in zip(tagged_seq, test_run_base) if i != j
            ]
            print("Testing model performance on test data : ")
            print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
            print(
                bcolors.OKGREEN
                + bcolors.BOLD
                + "Time taken in seconds: "
                + bcolors.ENDC
                + str(difference)
            )
            print(
                bcolors.OKGREEN
                + bcolors.BOLD
                + "Accuracy"
                + bcolors.ENDC
                + " for this sentence: ",
                accuracy * 100,
                "%",
            )
            print(bcolors.WARNING + "-" * 100 + bcolors.ENDC)
            print("Incorrectly tagged words: ")
            print(bcolors.WARNING + "-" * 100 + bcolors.ENDC)
            # pprint.pprint(incorrectly_tagged_words)
            print(
                "{:>20} | {:>20} | {:>20}".format(
                    "Token", "Predicted tag", "Correct tag"
                )
            )
            print(bcolors.FAIL + "-" * 100 + bcolors.ENDC)
            for item in incorrectly_tagged_words:
                print(
                    "{:>20} | {:>20} | {:>20}".format(
                        item[0].token, item[0].POS, item[1].POS
                    )
                )
                # print(item[0].token, item[0].POS, item[1].POS)
            print("-" * 100)

        total_accuracy = total_correct / total_tagged_seq
        print(bcolors.OKCYAN + "-" * 100 + bcolors.ENDC)
        print(
            bcolors.OKGREEN
            + bcolors.BOLD
            + "Overall test accuracy : "
            + str(total_accuracy * 100)
            + "%"
            + bcolors.ENDC
        )
        print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)


if __name__ == "__main__":

    if DEMO:
        hmm = HMM("demo_corpus.txt")
        hmm.fit(printing=True)
        hmm.test_performance()
    else:
        hmm = HMM("brown_corpus.txt")
        hmm.fit(printing=False)
        hmm.test()
