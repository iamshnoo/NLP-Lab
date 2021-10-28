from utils import nltk_testing_data, nltk_training_data

from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from contextlib import redirect_stdout


with open("outputs/nltk_perplexity.txt", "w") as data_file:
    with redirect_stdout(data_file):
        N = 2
        a = nltk_training_data(N)
        b = nltk_testing_data(N)
        train_data, padded_vocab = padded_everygram_pipeline(N, a)
        test_data, _ = padded_everygram_pipeline(N, b)

        ## NLTK IMPLEMENTATION of ADD ONE
        model = Laplace(N)
        model.fit(train_data, padded_vocab)

        for i, test in enumerate(test_data):
            print("{:>5}{:>30}".format(i, round(model.perplexity(test), 2)))
