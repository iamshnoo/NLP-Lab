from nltk.tokenize import word_tokenize
import string
from process import ProcessData


def process_sentence(sentence):
    y = word_tokenize(sentence)
    return [word.lower() for word in y if word not in string.punctuation]


def testing_data(N):
    x = ProcessData("testing_data.txt", N)
    x.modify_data()
    return x.get_lines()


def training_data(N):
    x = ProcessData("training_data.txt", N)
    x.modify_data()
    return x.get_lines()


def nltk_testing_data(N):
    x = ProcessData("testing_data.txt", N)
    x.modify_data()
    y = x.get_lines()

    processed_sentences = []
    for i in y:
        z = process_sentence(i)
        if z:
            processed_sentences.append(z)

    return processed_sentences


def nltk_training_data(N):
    x = ProcessData("training_data.txt", N)
    x.modify_data()
    y = x.get_lines()

    processed_sentences = []
    for i in y:
        z = process_sentence(i)
        if z:
            processed_sentences.append(z)

    return processed_sentences
