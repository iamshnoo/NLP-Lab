import random
import time

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
        self.corpus = corpus
        self.data = Dataset(corpus)
        self.tag_indices = dict(zip(self.data.tagset, range(len(self.data.tagset))))
        self.token_indices = dict(zip(self.data.vocab, range(len(self.data.vocab))))

        self.pi = [0 for _ in range(len(self.tag_indices))]
        self.transition = [
            [0 for i in range(len(self.tag_indices))]
            for j in range(len(self.tag_indices))
        ]
        self.emission = [
            [0 for i in range(len(self.token_indices))]
            for j in range(len(self.tag_indices))
        ]

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

    def smooth_emission_prob(self, emission, tag, token):
        total_tokens_from_tag = self.token_count[tag]
        emission_count = (
            emission[tag][token] if token in emission.index else 0
        ) * total_tokens_from_tag
        vocab_size = len(emission.index)

        return (emission_count + 1) / (total_tokens_from_tag + vocab_size)

    def test(self, test_sentence=None):
        # Checking model performance on a single test sentence
        test = (
            "If the circumstances are faced frankly it is reasonable to expect this to be true."
            if not test_sentence
            else test_sentence
        )
        test_tagged_words = word_tokenize(test)
        tagged_seq = self.most_probable_tags(test_tagged_words, print_vmat=True)
        print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
        print("Testing model performance on random test sentence : ")
        print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
        print("Sentence :", test)
        print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
        print("{:>20} | {:>20}".format("Token", "Predicted POS tag"))
        print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
        for item in tagged_seq:
            print("{:>20} | {:>20}".format(item.token, item.POS))
        print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)

    def test_performance(self):
        # Checking model performance on test data
        # for brown corpus, check on all test sentences except 10
        # for demo corpus, check on all test sentences
        total_correct = 0
        total_tagged_seq = 0
        for i, sentence in enumerate(self.data.test_sentences):
            # these sentences are too long, and hence the vmat matrix values
            # zero out due to floating point precision limitations.
            if (
                self.corpus == "brown_corpus.txt"
                and i
                not in [426, 1168, 1228, 3830, 5089, 8417, 8564, 8894, 9372, 10281]
            ) or (self.corpus == "demo_corpus.txt"):
                print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
                print(
                    bcolors.OKGREEN + bcolors.BOLD + "Test Sentence ",
                    i + 1,
                    " : " + bcolors.ENDC,
                )
                words = [item.token for item in sentence]
                print(words)
                print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
                test_run_base = [word for word in sentence]
                test_tagged_words = [word.token for word in sentence]

                start = time.time()
                tagged_seq = self.most_probable_tags(
                    test_tagged_words, print_vmat=False
                )
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

    def pi_counts(self):
        for sentence in self.data.train_sentences:
            for i, item in enumerate(sentence):
                # pi is calculated for start words of sentences only
                if i == 0:
                    self.pi[self.tag_indices[item.POS]] += 1

    def transition_counts(self):
        for sentence in self.data.train_sentences:
            for i, item in enumerate(sentence):
                if i != 0:
                    self.transition[self.tag_indices[sentence[i - 1][1]]][
                        self.tag_indices[item.POS]
                    ] += 1

    def emission_counts(self):
        for sentence in self.data.train_sentences:
            for i, item in enumerate(sentence):
                self.emission[self.tag_indices[item.POS]][
                    self.token_indices[item.token]
                ] += 1

    def counts(self):
        self.pi_counts()
        self.transition_counts()
        self.emission_counts()

    def pi_probs(self):
        pi_sum = sum(self.pi)
        for i in range(len(self.pi)):
            self.pi[i] /= pi_sum
        self.pi_df = pd.DataFrame(self.pi, index=self.data.tagset).transpose()

    def transition_probs(self):
        for tag, row in self.tag_indices.items():
            transition_sum = sum(self.transition[row])
            for i in range(len(self.tag_indices)):
                self.transition[row][i] = (
                    self.transition[row][i] / transition_sum
                    if transition_sum != 0
                    else 0
                )
        self.transition_df = pd.DataFrame(
            self.transition, index=self.data.tagset, columns=self.data.tagset
        ).transpose()

    def emission_probs(self):
        for tag, row in self.tag_indices.items():
            emission_sum = sum(self.emission[row])
            for i in range(len(self.token_indices)):
                self.emission[row][i] = (
                    self.emission[row][i] / emission_sum if emission_sum != 0 else 0
                )
            self.token_count[tag] = emission_sum
        self.emission_df = pd.DataFrame(
            self.emission, index=self.data.tagset, columns=self.data.vocab
        ).transpose()

    def probs(self):
        self.pi_probs()
        self.transition_probs()
        self.emission_probs()

    def print_hmm_internals(self):
        print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
        print(bcolors.OKGREEN + bcolors.BOLD + "Pi : " + bcolors.ENDC)
        print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
        print(self.pi_df)
        print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
        print(bcolors.OKGREEN + bcolors.BOLD + "Transition matrix : " + bcolors.ENDC)
        print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
        print(self.transition_df)
        print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
        print(bcolors.OKGREEN + bcolors.BOLD + "Emission matrix : " + bcolors.ENDC)
        print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
        print(self.emission_df)
        print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)

    def learn(self, print_hmm_params=True):

        self.counts()
        self.probs()

        if print_hmm_params:
            self.print_hmm_internals()

        return (self.pi_df, self.transition_df, self.emission_df)

    @staticmethod
    def reverseString(input_string):
        """Utility method to reverse a string."""
        start = stop = None
        step = -1
        reverse_slice = slice(start, stop, step)
        return input_string[reverse_slice]

    # Implements Viterbi algorithm with smoothing for unknown words
    def most_probable_tags(self, words, hmm_parameter=None, print_vmat=True):
        # sourcery no-metrics
        if hmm_parameter:
            (pi, transition, emission) = hmm_parameter
        else:
            (pi, transition, emission) = (
                self.pi_df,
                self.transition_df,
                self.emission_df,
            )

        # this is a slightly modified vmat matrix that stores both probability
        # values and also the previous state from which the current state has
        # been reached for each state.
        vmat = pd.DataFrame(
            [[None for _i in self.training_tagset] for _j in words],
            index=[str(i) for i in range(len(words))],
            columns=self.training_tagset,
        ).transpose()

        # this is the first column of vmat
        # prob values are obtained by multiplying pi values with emission values
        for state in self.training_tagset:
            vmat[str(0)][state] = {
                "prob": pi[state][0]
                * self.smooth_emission_prob(emission, state, words[0]),
                "prev": None,
            }

        # for each column of vmat, loop through all possible tags to find most
        # probable transition.
        for i in range(1, len(words)):
            for state in self.training_tagset:
                max_transition_prob = (
                    vmat[str(i - 1)][self.training_tagset[0]]["prob"]
                    * transition[self.training_tagset[0]][state]
                )
                previous_state_selected = self.training_tagset[0]
                for previous_state in self.training_tagset[1:]:
                    transition_prob = (
                        vmat[str(i - 1)][previous_state]["prob"]
                        * transition[previous_state][state]
                    )
                    if transition_prob > max_transition_prob:
                        max_transition_prob = transition_prob
                        previous_state_selected = previous_state

                max_prob = max_transition_prob * self.smooth_emission_prob(
                    emission, state, words[i]
                )
                vmat[str(i)][state] = {
                    "prob": max_prob,
                    "prev": previous_state_selected,
                }

        # backtrack using the previous state information stored in vmat
        states = []
        most_probable_state = None
        max_prob = 0.0

        # first find the last tag of the sequence
        # this is simply the tag corresponding to the maximum probability in the
        # last column of vmat
        for state, data in vmat[str(len(words) - 1)].items():
            if data["prob"] > max_prob:
                max_prob = data["prob"]
                most_probable_state = state
        states.append(most_probable_state)
        previous = most_probable_state

        # find the other tags using backpointers
        for i in range(len(words) - 2, -1, -1):
            states.append(vmat[str(i + 1)][previous]["prev"])
            previous = vmat[str(i + 1)][previous]["prev"]

        if print_vmat:
            print("vmat matrix : ")
            print("-" * 100)
            print(vmat)
            print("-" * 100)

        return [
            Word(token, POS) for token, POS in zip(words, self.reverseString(states))
        ]


if __name__ == "__main__":

    start = time.time()
    hmm = HMM("demo_corpus.txt") if DEMO else HMM("brown_corpus.txt")
    hmm.learn()
    hmm.test()
    # hmm.test_performance()
    # end = time.time()
    # difference = end - start
    # print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
    # print(
    #     bcolors.OKGREEN
    #     + bcolors.BOLD
    #     + "Time taken in seconds for execution: "
    #     + bcolors.ENDC
    #     + str(difference)
    # )
    # print(bcolors.OKBLUE + "-" * 100 + bcolors.ENDC)
