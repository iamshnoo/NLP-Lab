import pprint
import time
import random
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

Word = namedtuple("Word", "token POS")

if __name__ == "__main__":

    # reading the Brown tagged sentences
    # nltk_data = list(nltk.corpus.brown.tagged_sents(tagset='universal'))

    def load_corpus(filename):
        with open(filename, "r") as f:
            sentence_lines = f.read().split("\n")
        sentences = []
        for line in sentence_lines:
            words = line.split(" ")
            tuples = []
            for word in words:
                if not word:
                    break
                token, POS = word.split("=")
                tuples.append(Word(token, POS))
            if tuples:
                sentences.append(tuples)
        return sentences

    def read_tags(filename):
        """Read a list of word tag classes"""
        with open(filename, "r") as f:
            tags = f.read().split("\n")
        return set(tags)

    tagfile = "tags_universal.txt"
    datafile = "demo_corpus.txt"
    tagset = read_tags(tagfile)
    sentences = load_corpus(datafile)
    tagged_words = [tup for sent in sentences for tup in sent]
    vocab = set([word.token for sentence in sentences for word in sentence])

    random.seed(42)
    # split data into training and testing set
    train_sentences, test_sentences = train_test_split(
        sentences, train_size=0.9, test_size=0.1, random_state=42
    )

    # create list of train and test tagged words
    train_tagged_words = [tup for sent in train_sentences for tup in sent]
    test_tagged_words = [tup.token for sent in test_sentences for tup in sent]

    # training vocab and tagset
    training_tagset = set([word.POS for word in train_tagged_words])
    training_vocab = set([word.token for word in train_tagged_words])

    # The initial tag probabilities π(ti) for 1<i< n, where π(ti) is the probability that a
    # sentence begins with tag ti.
    sentence_start_words = []
    for sentence in train_sentences:
        for i, word in enumerate(sentence):
            if i == 0:
                sentence_start_words.append(word)

    pi = []
    total_tag = len([word.POS for word in train_tagged_words])
    for t in training_tagset:
        each_tag = [word.POS for word in sentence_start_words if word.POS == t]
        pi.append((t, len(each_tag) / total_tag))

    print(pi)

    # The transition probabilities a(ti →tj) for 1<(i, j)<n where a(ti →tj) is the
    # probability that tag tj occurs after tag ti.
    def P_tj_given_ti(tj, ti, train_bag=train_tagged_words):
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

    # creating transition matrix
    # no. of cols = no. of rows = no. of hidden states = no. of tags
    transition_matrix = np.zeros(
        (len(training_tagset), len(training_tagset)), dtype="float32"
    )

    # each column is tj, each row is ti
    # M(i, j) represents P(tj given ti)
    for i, ti in enumerate(list(training_tagset)):
        for j, tj in enumerate(list(training_tagset)):
            transition_matrix[i, j] = P_tj_given_ti(tj, ti)

    # convert the matrix to a df
    transition_df = pd.DataFrame(
        transition_matrix, columns=list(training_tagset), index=list(training_tagset)
    )

    print(transition_df)

    # The emission probabilities b(ti →wj) for 1<i< n and 1<j< n where b(ti →wj) is the
    # probability that token wj is generated given tag ti.
    def P_word_given_tag(word, tag, train_bag=train_tagged_words):
        taglist = [item for item in train_bag if item.POS == tag]
        tag_count = len(taglist)
        w_in_tag = [item.token for item in taglist if item.token == word]
        word_count_given_tag = len(w_in_tag)
        return word_count_given_tag / tag_count

    # creating emission matrix
    # no. of rows = no. of hidden states = no. of tags
    # no. of cols = no. of visible states = no. of words in vocabulary
    emission_matrix = np.zeros(
        (len(training_tagset), len(training_vocab)), dtype="float32"
    )

    # each column is wj, each row is ti
    # M(i, j) represents P(wj given ti)
    for i, ti in enumerate(list(training_tagset)):
        for j, wj in enumerate(list(training_vocab)):
            emission_matrix[i, j] = P_word_given_tag(wj, ti)

    # convert the matrix to a df
    emission_df = pd.DataFrame(
        emission_matrix, columns=list(training_vocab), index=list(training_tagset)
    )

    print(emission_df)

    # Vanilla Viterbi Algorithm
    def most_probable_tags(words, train_bag=train_tagged_words):
        state = []
        T = list({word.POS for word in train_bag})
        print(T)

        for key, word in enumerate(words):
            # initialise list of probability column for a given observation
            p = []
            for tag in T:
                if key == 0:
                    transition_p = transition_df.loc[".", tag]
                else:
                    transition_p = transition_df.loc[state[-1], tag]

                # compute emission and state probabilities
                emission_p = P_word_given_tag(word, tag)
                state_probability = emission_p * transition_p
                p.append(state_probability)

            p_max = max(p)
            # getting state for which probability is maximum
            state_max = T[p.index(p_max)]
            state.append(state_max)

        return [Word(token, POS) for token, POS in zip(words, state)]

    # Checking model performance on test set
    test_run_base = [word for sentence in test_sentences for word in sentence]
    test_tagged_words = [word.token for sentence in test_sentences for word in sentence]
    start = time.time()
    tagged_seq = most_probable_tags(test_tagged_words)
    end = time.time()
    difference = end - start
    correct = [i for i, j in zip(tagged_seq, test_run_base) if i == j]
    accuracy = len(correct) / len(tagged_seq)
    incorrectly_tagged_words = [
        (i, j) for i, j in zip(tagged_seq, test_run_base) if i != j
    ]
    print("Time taken in seconds: ", difference)
    print("Vanilla Viterbi Algorithm Accuracy: ", accuracy * 100)
    print("Incorrectly tagged words, Corresponding correct tags : ")
    pprint.pprint(incorrectly_tagged_words)
