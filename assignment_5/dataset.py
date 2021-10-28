import pprint
import random
from collections import namedtuple

from sklearn.model_selection import train_test_split

Word = namedtuple("Word", "token POS")


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
    return sorted(set(tags))


class Dataset(
    namedtuple(
        "_Dataset",
        "tagset sentences tagged_words vocab train_sentences test_sentences",
    )
):
    def __new__(
        cls,
        datafile,
        tagfile="tags_universal.txt",
        split=0.8,
        seed=42,
    ):
        tagset = read_tags(tagfile)
        sentences = load_corpus(datafile)
        tagged_words = [tup for sent in sentences for tup in sent]
        vocab = sorted({word.token for sentence in sentences for word in sentence})

        # split data into training and testing set
        if seed is not None:
            random.seed(seed)

        train_sentences, test_sentences = train_test_split(
            sentences, train_size=split, random_state=seed
        )

        return super().__new__(
            cls, tagset, sentences, tagged_words, vocab, train_sentences, test_sentences
        )

    def __len__(self):
        return len(self.sentences)


if __name__ == "__main__":
    data = Dataset("demo_corpus.txt")
    print(len(data), len(data.train_sentences), len(data.test_sentences))

    for sentence in data.test_sentences:
        pprint.pprint(sentence)
        print("-" * 100)
