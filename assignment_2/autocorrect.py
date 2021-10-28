import functools
import re
from collections import Counter
from contextlib import redirect_stdout
from typing import Any, List, Optional

from med import MinimumEditDistance


class AutoCorrect:
    def __init__(self, word_corpus: str) -> None:
        """Creates an instance for the class."""
        try:
            with open(word_corpus, "r") as inputFile:
                self.__word_corpus = inputFile.read()
                self.load_corpus()
        except IOError:
            print("Couldn't read input corpus.")
            exit(1)

    @staticmethod
    def tokenize(text: str) -> List:
        """List all the word tokens (consecutive letters) in a text.
        Normalize to lowercase."""
        return re.findall("[a-z]+", text.lower())

    @functools.lru_cache(maxsize=100)
    def load_corpus(self) -> None:
        """Tokenize the corpus and load it into a Counter."""
        self.CORPUS = self.tokenize(self.__word_corpus)
        self.COUNTS = Counter(self.CORPUS)
        self.UNIQUE_WORDS = sorted(set(self.CORPUS))

    def print_info(
        self, candidates: List, edit_distances: List, probabilites: List
    ) -> None:
        """Helper function for printing viable candidates, their frequencies and
        edit_distances when use_frequency=True.
        """
        print(
            "\nUsing a dictionary of ",
            len(self.UNIQUE_WORDS),
            " unique words \nand a total of ",
            len(self.CORPUS),
            " words to predict the most probable word...\n",
        )

        l = list(zip(candidates, probabilites, edit_distances))

        # sort by frequency
        l = sorted(l, key=lambda t: t[1], reverse=True)

        # sort by edit distance
        l = sorted(l, key=lambda t: t[2])

        print("List of viable candidates :")
        print("-" * 80)
        print("{:>20}{:>20}{:>20}".format("candidate", "frequency", "edit_distance"))
        print("-" * 80)
        for c, p, d in l:
            print("{:>20}{:>20}{:>20}".format(c, p, d))
        print("-" * 80)

    def autocorrect(
        self,
        word: str,
        threshold: Optional[int] = 2,
        use_frequency: Optional[bool] = True,
    ) -> Any:
        """
        Given a word w, find the most likely correction c = correct(w).

        Approach: Try all candidate words c that are known words that are 'near'
        w. Choose the most 'likely' one.

        To balance near and likely, in a trivial way: Measure nearness by
        edit distance <= threshold, and choose the most likely word from the
        given text by frequency.

        Reference :
        http://nbviewer.jupyter.org/url/norvig.com/ipython/How%20to%20Do%20Things%20with%20Words.ipynb
        """
        word = word.lower()

        # if the word is in the loaded corpus,
        if word in self.UNIQUE_WORDS:
            print("No corrections required.")
            return word

        med = MinimumEditDistance(
            insertion_cost=1, deletion_cost=1, substitution_cost=2
        )

        candidates = []
        edit_distances = []
        for candidate in self.UNIQUE_WORDS:
            weightedWagnerFischerResults = med.weightedWagnerFischer(word, candidate)
            distance = weightedWagnerFischerResults["edit_distance"]
            if distance <= threshold:
                candidates.append(candidate)
                edit_distances.append(distance)

        if not candidates:
            print("No viable correction found for given word.")
            return word

        if not use_frequency:
            return candidates

        probabilites = [self.COUNTS[possible] for possible in candidates]
        l = list(zip(candidates, probabilites, edit_distances))

        # prefer edits that are 1 edit away
        edit1 = [c for c, p, d in l if d == 1]
        if edit1:
            viable_candidate = max(edit1, key=self.COUNTS.get)

        # if there are no words 1 edit away, consider all words
        else:
            viable_candidate = max(candidates, key=self.COUNTS.get)

        return viable_candidate, candidates, edit_distances, probabilites


if __name__ == "__main__":
    autocorrection_tool = AutoCorrect("shakespeare.txt")
    test1 = "sui"
    (
        x,
        candidates,
        edit_distances,
        probabilites,
    ) = autocorrection_tool.autocorrect(test1, threshold=2, use_frequency=True)

    test2 = "sui"
    x2 = autocorrection_tool.autocorrect(test2, threshold=2, use_frequency=False)

    with open("autocorrect-output.txt", "w") as f1:
        with redirect_stdout(f1):
            autocorrection_tool.print_info(candidates, edit_distances, probabilites)
            print("Given word : ", test1)
            print("Chosen candidate(s) : ", x)
            print()
            print("-" * 80)
            print()
            print("Without considering any dictionary : \n")
            print("Given word : ", test2)
            print("Chosen candidate(s) : ", x2)
            print()
            print()
            print("-" * 80)
            print()
