import re
from collections import namedtuple

from nltk import Nonterminal, induce_pcfg
from nltk.corpus import treebank

import random
from sklearn.model_selection import train_test_split
from nltk.tokenize.treebank import TreebankWordDetokenizer

ROOT_TOKEN = "S"


class Dataset(
    namedtuple(
        "_Dataset",
        "fileids productions pcfg_grammar test_sentences",
    )
):
    def __new__(
        cls,
        seed=42,
        split=0.99,
        pcfg_file="pcfg.txt",
        test_set="test_sentences.txt",
        nltk_test_set="nltk_test.txt",
    ):
        fileids = treebank.fileids()[:100]
        fileids.remove("wsj_0056.mrg")

        # split data into training and testing set
        if seed is not None:
            random.seed(seed)

        train_fileids, test_fileids = train_test_split(
            fileids, train_size=split, random_state=seed
        )

        productions = []
        for id in train_fileids:
            for tree in treebank.parsed_sents(id):
                tree.collapse_unary(collapsePOS=True)
                tree.chomsky_normal_form()
                productions += tree.productions()

        pcfg_grammar = induce_pcfg(Nonterminal(ROOT_TOKEN), productions)

        with open(nltk_test_set, "w") as f:
            for id in test_fileids:
                for tree in treebank.parsed_sents(id):
                    item = tree.leaves()
                    sentence = TreebankWordDetokenizer().detokenize(item)
                    line = sentence + "\n"
                    f.write(line)

        with open(pcfg_file, "w") as grammar_file:
            for production in pcfg_grammar.productions():
                line = (
                    re.sub(r" ?\[[^)]+\]", "", str(production))
                    + " "
                    + str(production.prob())
                    + "\n"
                )
                grammar_file.write(line)

        with open(test_set, "r") as test_file:
            test_sentences = [line for line in test_file]

        return super().__new__(cls, fileids, productions, pcfg_grammar, test_sentences)

    def __len__(self):
        return len(self.fileids)


if __name__ == "__main__":
    data = Dataset()
