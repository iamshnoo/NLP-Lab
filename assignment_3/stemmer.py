from utils import StemmerUtility
from constants import affix_rules

# Implements https://tartarus.org/martin/PorterStemmer/def.txt
class NaiveStemmer(object):
    def __init__(self) -> None:
        self.stemmer_utils = StemmerUtility()

    def __step1a(self, word: str) -> str:

        for key, value in affix_rules["step1a"].items():
            if word.endswith(key):
                word = word[: word.rfind(key)] + value
                break

        return word

    def __step1b(self, word: str) -> str:

        flag = False
        for key, value in affix_rules["step1b"].items():
            if word.endswith(key):
                base = word[: word.rfind(key)]
                if key == "eed":
                    if self.stemmer_utils.get_m(base) > 0:
                        word = base + value
                    break
                if key in ["ed", "ing"]:
                    if self.stemmer_utils.contains_vowel(base):
                        word = base + value
                        flag = True
                    break

        if flag:
            if word.endswith("at"):
                word += "e"
            elif word.endswith("bl"):
                word += "e"
            elif word.endswith("iz"):
                word += "e"
            elif (
                self.stemmer_utils.is_consonant(word, -1)
                and self.stemmer_utils.is_consonant(word, -2)
                and word[-1] not in "lsz"
            ):
                word = word[:-1]
            elif self.stemmer_utils.get_m(word) == 1 and self.stemmer_utils.CVC(word):
                word += "e"
        return word

    def __step1c(self, word: str) -> str:

        for key, value in affix_rules["step1c"].items():
            if word.endswith(key):
                base = word[: word.rfind(key)]
                if self.stemmer_utils.contains_vowel(base):
                    word = base + value
            return word

    def __step2(self, word: str) -> str:

        for key, value in affix_rules["step2"].items():
            if word.endswith(key):
                base = word[: word.rfind(key)]
                if self.stemmer_utils.get_m(base) > 0:
                    word = base + value
                break
        return word

    def __step3(self, word: str) -> str:

        for key, value in affix_rules["step3"].items():
            if word.endswith(key):
                base = word[: word.rfind(key)]
                if self.stemmer_utils.get_m(base) > 0:
                    word = base + value
                break

        return word

    def __step4(self, word: str) -> str:

        for key, value in affix_rules["step4"].items():
            if word.endswith(key):
                base = word[: word.rfind(key)]
                if self.stemmer_utils.get_m(base) > 1:
                    word = base + value
                break

        if word.endswith("ion"):
            base = word[: word.rfind("ion")]
            if self.stemmer_utils.get_m(base) > 1 and base[-1] in "st":
                word = base

        return word

    def __step5a(self, word: str) -> str:

        for key, value in affix_rules["step5a"].items():
            if word.endswith(key):
                base = word[: word.rfind(key)]
                if (self.stemmer_utils.get_m(base) > 1) or (
                    self.stemmer_utils.get_m(base) == 1
                    and not self.stemmer_utils.CVC(base)
                ):
                    word = base + value
            return word

    def __step5b(self, word: str) -> str:

        for key, value in affix_rules["step5b"].items():
            if word.endswith(key):
                base = word[: word.rfind(key)]
                if self.stemmer_utils.get_m(word) > 1:
                    word = base + value

        return word

    def stem(self, word: str) -> str:

        word = self.__step5b(
            self.__step5a(
                self.__step4(
                    self.__step3(
                        self.__step2(self.__step1c(self.__step1b(self.__step1a(word))))
                    )
                )
            )
        )
        return word


if __name__ == "__main__":
    stemmer = NaiveStemmer()
    word = "altered"
    print("Input word :", word)
    print("Output stem :", stemmer.stem(word))
