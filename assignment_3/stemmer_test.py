import unittest
from nltk.stem import PorterStemmer
from prefix_removal import AdvancedStemmer
from stemmer import NaiveStemmer
from contextlib import redirect_stdout
from constants import *


class TestAffixStemmer(unittest.TestCase):
    def test_step1a(self) -> None:
        self._extracted_from_test_step1a_2(
            "", "Naive Stemmer (Suffix Removal) Testing : "
        )

        stemmer = NaiveStemmer()
        nltk_stemmer = PorterStemmer()
        self._extracted_from_test_step1a_2("-", "Step 1a Testing : ")
        print("{:>20}{:>20}{:>20}".format("Word", "Naive stem", "NLTK stem"))
        print("-" * 80)
        for word in step1a_words:
            naive_stem = stemmer.stem(word)
            nltk_stem = nltk_stemmer.stem(word)
            print("{:>20}{:>20}{:>20}".format(word, naive_stem, nltk_stem))
            message = "Comparing " + naive_stem + " with " + nltk_stem + " failed!"
            self.assertEqual(naive_stem, nltk_stem, message)
        print("-" * 80)

    def _extracted_from_test_step1a_2(self, arg0, arg1):
        print(arg0 * 80)
        print(arg1)
        print(arg0 * 80)

    def test_step1b(self) -> None:
        stemmer = NaiveStemmer()
        nltk_stemmer = PorterStemmer()
        print("-" * 80)
        print("Step 1b Testing : ")
        print("-" * 80)
        print("{:>20}{:>20}{:>20}".format("Word", "Naive stem", "NLTK stem"))
        print("-" * 80)
        for word in step1b_words:
            naive_stem = stemmer.stem(word)
            nltk_stem = nltk_stemmer.stem(word)
            print("{:>20}{:>20}{:>20}".format(word, naive_stem, nltk_stem))
            message = "Comparing " + naive_stem + " with " + nltk_stem + " failed!"
            self.assertEqual(naive_stem, nltk_stem, message)
        print("-" * 80)

    def test_step1c(self) -> None:
        stemmer = NaiveStemmer()
        nltk_stemmer = PorterStemmer()
        print("-" * 80)
        print("Step 1c Testing : ")
        print("-" * 80)
        print("{:>20}{:>20}{:>20}".format("Word", "Naive stem", "NLTK stem"))
        print("-" * 80)
        for word in step1c_words:
            naive_stem = stemmer.stem(word)
            nltk_stem = nltk_stemmer.stem(word)
            print("{:>20}{:>20}{:>20}".format(word, naive_stem, nltk_stem))
            message = "Comparing " + naive_stem + " with " + nltk_stem + " failed!"
            self.assertEqual(naive_stem, nltk_stem, message)
        print("-" * 80)

    def test_step2(self) -> None:
        stemmer = NaiveStemmer()
        nltk_stemmer = PorterStemmer()
        print("-" * 80)
        print("Step 2 Testing : ")
        print("-" * 80)
        print("{:>20}{:>20}{:>20}".format("Word", "Naive stem", "NLTK stem"))
        print("-" * 80)
        for word in step2_words:
            naive_stem = stemmer.stem(word)
            nltk_stem = nltk_stemmer.stem(word)
            print("{:>20}{:>20}{:>20}".format(word, naive_stem, nltk_stem))
            message = "Comparing " + naive_stem + " with " + nltk_stem + " failed!"
            self.assertEqual(naive_stem, nltk_stem, message)
        print("-" * 80)

    def test_step3(self) -> None:
        stemmer = NaiveStemmer()
        nltk_stemmer = PorterStemmer()
        print("-" * 80)
        print("Step 3 Testing : ")
        print("-" * 80)
        print("{:>20}{:>20}{:>20}".format("Word", "Naive stem", "NLTK stem"))
        print("-" * 80)
        for word in step3_words:
            naive_stem = stemmer.stem(word)
            nltk_stem = nltk_stemmer.stem(word)
            print("{:>20}{:>20}{:>20}".format(word, naive_stem, nltk_stem))
            message = "Comparing " + naive_stem + " with " + nltk_stem + " failed!"
            self.assertEqual(naive_stem, nltk_stem, message)
        print("-" * 80)

    def test_step4(self) -> None:
        stemmer = NaiveStemmer()
        nltk_stemmer = PorterStemmer()
        print("-" * 80)
        print("Step 4 Testing : ")
        print("-" * 80)
        print("{:>20}{:>20}{:>20}".format("Word", "Naive stem", "NLTK stem"))
        print("-" * 80)
        for word in step4_words:
            naive_stem = stemmer.stem(word)
            nltk_stem = nltk_stemmer.stem(word)
            print("{:>20}{:>20}{:>20}".format(word, naive_stem, nltk_stem))
            message = "Comparing " + naive_stem + " with " + nltk_stem + " failed!"
            self.assertEqual(naive_stem, nltk_stem, message)
        print("-" * 80)

    def test_step5a(self) -> None:
        stemmer = NaiveStemmer()
        nltk_stemmer = PorterStemmer()
        print("-" * 80)
        print("Step 5a Testing : ")
        print("-" * 80)
        print("{:>20}{:>20}{:>20}".format("Word", "Naive stem", "NLTK stem"))
        print("-" * 80)
        for word in step5a_words:
            naive_stem = stemmer.stem(word)
            nltk_stem = nltk_stemmer.stem(word)
            print("{:>20}{:>20}{:>20}".format(word, naive_stem, nltk_stem))
            message = "Comparing " + naive_stem + " with " + nltk_stem + " failed!"
            self.assertEqual(naive_stem, nltk_stem, message)
        print("-" * 80)

    def test_step5b(self) -> None:
        stemmer = NaiveStemmer()
        nltk_stemmer = PorterStemmer()
        print("-" * 80)
        print("Step 5b Testing : ")
        print("-" * 80)
        print("{:>20}{:>20}{:>20}".format("Word", "Naive stem", "NLTK stem"))
        print("-" * 80)
        for word in step5b_words:
            naive_stem = stemmer.stem(word)
            nltk_stem = nltk_stemmer.stem(word)
            print("{:>20}{:>20}{:>20}".format(word, naive_stem, nltk_stem))
            message = "Comparing " + naive_stem + " with " + nltk_stem + " failed!"
            self.assertEqual(naive_stem, nltk_stem, message)
        print("-" * 80)

    def test_prefix_removal(self) -> None:
        stemmer = AdvancedStemmer()
        nltk_stemmer = PorterStemmer()
        print("" * 80)
        print("Prefix Removal Stemmer Testing : ")
        print("" * 80)
        print("-" * 80)
        print("{:>20}{:>20}{:>35}".format("Word", "Advanced stem", "NLTK stem"))
        print("-" * 80)
        for word in advanced_stemmer_test_words:
            naive_stem = stemmer.stem(word)
            nltk_stem = nltk_stemmer.stem(word)
            print("{:>20}{:>20}{:>35}".format(word, naive_stem, nltk_stem))
        print("-" * 80)


if __name__ == "__main__":
    with open("stemmer_test-output.txt", "w") as f1:
        with redirect_stdout(f1):
            unittest.main()
