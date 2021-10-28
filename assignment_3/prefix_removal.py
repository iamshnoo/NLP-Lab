from stemmer import NaiveStemmer
from constants import english_prefixes
from nltk.corpus import words
from nltk.corpus import wordnet as wn

english_words = list(wn.words()) + words.words()


class AdvancedStemmer(object):
    def __init__(self) -> None:
        self.stemmer = NaiveStemmer()
        self.prefixes = english_prefixes
        self.words = list(wn.words()) + words.words()
        self.prefixes.sort(key=len, reverse=True)

    def remove_prefix(self, word: str) -> str:
        for i in range(len(self.prefixes)):
            prefix = self.prefixes[i]
            if word.startswith(prefix):
                word_without_prefix = word[len(prefix) :]
                if word_without_prefix.lower() in self.words:
                    return word_without_prefix
        return word

    def stem(self, word: str) -> str:
        word_without_prefix = self.remove_prefix(word)
        return self.stemmer.stem(word_without_prefix)


if __name__ == "__main__":
    stemmer = AdvancedStemmer()
    word = "reload"
    print("Input word :", word)
    print("Prefix removed stem :", stemmer.stem(word))
