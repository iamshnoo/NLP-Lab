import random
import string
from contextlib import redirect_stdout
from typing import List, Optional

import nltk
from nltk.corpus import brown, stopwords
from nltk.tokenize import word_tokenize


class Basics:
    """ Assignment 1 Solution Class"""

    def __init__(
        self, path: str, language: str, stop_words: Optional[str] = None
    ) -> None:
        """Creates an object of the class with given parameters."""

        super().__init__()
        self.PATH = path
        self.language = language
        self.stop_words = stop_words

        try:
            with open(self.PATH, "r") as inputFile:
                self.raw_text = inputFile.read()
        except IOError:
            self.raw_text = None
            print("Couldn't read input file.")

    def __repr__(self) -> str:
        """Formalized string representation of this class."""

        return (
            self.__class__.__qualname__
            + "("
            + self.PATH
            + ","
            + self.language
            + ") object."
        )

    def add_line_numbers(self) -> None:
        """Add a line number to every non-empty line of input file and print it."""

        if self.raw_text is None:
            raise Exception("Couldn't read input file.")

        line_number = 1
        for line in self.raw_text.splitlines():

            # empty lines and whitespaces are ignored
            if not line.isspace() and line != "":
                print(line_number, "\t", line)
                line_number += 1

    def __tokenize(self, text: Optional[str] = None) -> List:
        """Normalizes input text by converting to lower case and then tokenizing
        it."""

        if text:
            return word_tokenize(text.lower())
        else:
            if self.raw_text is None:
                raise Exception("Couldn't read input file.")
            else:
                return word_tokenize(self.raw_text.lower())

    def __remove_punctuations(self, tokens: List) -> List:
        """Removes all possible punctuations from the list of tokens."""

        if not tokens:
            raise Exception("No tokens found.")
        # list of all English punctuations
        punctuations = set(string.punctuation)

        # this is the only Bengali punctuation which is not there in English
        if self.language == "bn":
            punctuations.add("।")

        # if at least one of the characters is not a punctuation, it is a word.
        # this removes all the single character punctuations from the list of
        # tokens.
        words = []
        for token in tokens:
            for char in token:
                if char not in punctuations:
                    words.append(token)
                    break

        return words

    def __clean_tokens(self, tokens: List) -> List:
        """Removes stopwords, contractions, conjunctions."""

        if not tokens:
            raise Exception("No tokens found.")
            # use english stopwords by default, otherwise use specified stopwords
        if self.stop_words is None:
            stop_words = set(stopwords.words("english"))
        else:
            stop_words = self.stop_words

            # remove stopwords from collection of tokens
        clean_tokens = [word for word in tokens if word not in stop_words]
        # remove contractions (eg. hasn't) and conjunctions (eg. on-campus)
        # the following list comprehension removes a token if any character in
        # the token is not an alphabet
        if self.language == "en":
            return [word for word in clean_tokens if word.isalpha()]

        return clean_tokens

    def vocab_size(self, text: Optional[str] = None) -> int:
        """The size of the vocabulary is the number of unique tokens in input
        text."""

        # use a set to represent the vocabulary, as a set only stores unique
        # elements
        if not text:
            if self.raw_text is None:
                raise Exception("Couldn't read input file.")
            vocabulary = set(self.__tokenize(self.raw_text))
        else:
            vocabulary = set(self.__tokenize(text))
        return len(vocabulary)

    def word_freq(self, word: str, section: str) -> int:
        """Computes frequency of input word in given section of Brown corpus."""

        # find all the words in the given section
        section_words = self.__remove_punctuations(
            list(brown.words(categories=section))
        )

        # find frequency of given word
        freq = 0
        for token in section_words:
            if token == word:
                freq += 1
        return freq

    def test_word_freq(self) -> None:
        """Test the word_freq method of the class by comparing with
        nltk.FreqDist."""

        # choose 5 random categories of brown corpus
        brown_categories = brown.categories()
        test_categories = random.sample(brown_categories, 5)

        print("-" * 26)
        print("{:>5} |{:>16}".format("word_freq", "nltk.FreqDist"))
        print("-" * 26)

        # choose 3 random words from each of the 5 random categories chosen above
        for category in test_categories:
            section_words = self.__remove_punctuations(
                list(brown.words(categories=category))
            )
            test_words = random.sample(section_words, 3)
            nltk_freq_dist = nltk.FreqDist(section_words)
            for word in test_words:
                # verify the correctness using assert statement
                assert self.word_freq(word, category) == nltk_freq_dist[word]

                # print the values as well for visual comparison
                print(
                    "{:>5}{:>15}".format(
                        self.word_freq(word, category), nltk_freq_dist[word]
                    )
                )

    def percent(self, word: str, text: Optional[str] = None) -> float:
        """Calculates how often a word occurs in a text as a percentage."""

        if not text:
            if self.raw_text is None:
                raise Exception("Couldn't read input file.")
            # calculate total number of words
            text_words = self.__remove_punctuations(self.__tokenize(self.raw_text))
        else:
            # calculate total number of words
            text_words = self.__remove_punctuations(self.__tokenize(text))
        total_count = len(text_words)

        # calculate frequency of given word
        frequency = text_words.count(word)
        # return percentage for word
        return (frequency / total_count) * 100

    def n_most_frequent(
        self, text: Optional[str] = None, num_words: Optional[int] = -1
    ) -> List:
        """Finds N most frequent words of text, except stopwords, contractions,
        conjunctions, punctuations."""

        # remove punctuations
        if text:
            tokens = self.__remove_punctuations(self.__tokenize(text))
        else:
            if self.raw_text is None:
                raise Exception("Couldn't read input file.")
            else:
                tokens = self.__remove_punctuations(self.__tokenize(self.raw_text))
        # remove stopwords, contractions, conjunctions
        cleaned_tokens = self.__clean_tokens(tokens)

        # calculate freq distribution of the tokens
        freq_dist = nltk.FreqDist(cleaned_tokens)

        # sort the frequency distribution in decreasing order of frequency
        sorted_freq_dist = sorted(freq_dist.items(), key=lambda item: -item[1])

        # if num_words is -1 (default), then return all word frequencies
        most_freq_words = []
        if num_words == -1:
            for i in range(len(sorted_freq_dist)):
                most_freq_words.append(sorted_freq_dist[i][0])
        else:
            for i in range(min(len(sorted_freq_dist), num_words)):
                most_freq_words.append(sorted_freq_dist[i][0])

        return most_freq_words

    def n_letter_words(
        self, text: Optional[str] = None, num_words: Optional[int] = 4
    ) -> List:
        """Finds all n letter words and prints them in decreasing order of
        frequency."""

        # find reverse-sorted frequency of all words
        if not text:
            if self.raw_text is not None:
                freq_sorted_words = self.n_most_frequent()
            else:
                raise Exception("Couldn't read input file.")
        else:
            if text:
                freq_sorted_words = self.n_most_frequent(text)
            else:
                raise Exception("Can't tokenize input.")

        # choose only n letter words from above output
        n_letter_words = []
        for word in freq_sorted_words:
            if len(word) == num_words:
                n_letter_words.append(word)

        return n_letter_words

    def words_occuring_n_times(self, count: Optional[int] = 3) -> List:
        """Finds all words that occur atleast n times in the Brown Corpus."""

        # all categories in brown corpus
        brown_categories = brown.categories()

        # aggregate all words from each category
        all_words = []
        for category in brown_categories:
            section_words = self.__remove_punctuations(
                list(brown.words(categories=category))
            )
            all_words.extend(section_words)

        # words from the list above, which have frequency >= count
        valid_words = []
        nltk_freq_dist = dict(nltk.FreqDist(all_words))
        for word, freq in nltk_freq_dist.items():
            if freq >= count:
                valid_words.append(word)

        return valid_words


def english() -> None:
    """Performs the required tasks for english input."""
    english_solution = Basics("sample-text-1.txt", "en")
    with open("en-output.txt", "w") as f1:
        with redirect_stdout(f1):
            print("-" * 100)
            english_solution.add_line_numbers()
            print("-" * 100)
            print("Vocabulary size : ", english_solution.vocab_size())
            print("-" * 100)
            english_solution.test_word_freq()
            print("-" * 100)
            word = "we"
            print(
                'Percentage of "',
                word,
                '" is : ',
                english_solution.percent(word),
                "%.",
            )
            print("-" * 100)
            print("10 most frequent words : ")
            print(english_solution.n_most_frequent(None, 10))
            print("-" * 100)
            print(
                "4 letter words in decreasing order of frequency from left to right: "
            )
            print(english_solution.n_letter_words())
            print("-" * 100)
            print("Words occuring thrice in Brown corpus : ")
            print(english_solution.words_occuring_n_times())
            print("-" * 100)


def bengali() -> None:
    """Performs the required tasks for bengali input."""
    stopwords_bn = set(open("stopwords-bn.txt").read().split())
    bengali_solution = Basics("sample-text-2.txt", "bn", stopwords_bn)
    with open("bn-output.txt", "w") as f2:
        with redirect_stdout(f2):
            _extracted_from_bengali_7(bengali_solution)


def _extracted_from_bengali_7(bengali_solution):
    print("-" * 100)
    bengali_solution.add_line_numbers()
    print("-" * 100)
    print("Vocabulary size : ", bengali_solution.vocab_size())
    print("-" * 100)
    word = "টাকা"
    print('Percentage of "', word, '" is : ', bengali_solution.percent(word), "%.")
    print("-" * 100)
    print("10 most frequent words : ")
    print(bengali_solution.n_most_frequent(None, 10))
    print("-" * 100)
    print("4 letter words in decreasing order of frequency from left to right: ")
    print(bengali_solution.n_letter_words())
    print("-" * 100)


if __name__ == "__main__":
    english()
    bengali()
