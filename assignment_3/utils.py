from constants import VOWELS


class StemmerUtility(object):

    # checks if i-th letter of word is a consonant
    def is_consonant(self, word, i):
        return (
            False
            if word[i] == "y" and word[i - 1] not in VOWELS
            else word[i] not in VOWELS
        )

    # checks if word contains any vowels
    def contains_vowel(self, word):
        vowels = [i for i in word if i in VOWELS]
        return len(vowels) > 0

    # converts word into the sequence [C](VC)^m[V]
    def __get_porter_form(self, word):
        form = []
        for i in range(len(word)):
            if i == 0:
                if self.is_consonant(word, i):
                    form.append("C")
                else:
                    form.append("V")
            else:
                if self.is_consonant(word, i) and form[-1] == "C":
                    pass
                elif self.is_consonant(word, i) and form[-1] == "V":
                    form.append("C")
                elif (self.is_consonant(word, i)) or form[-1] != "V":
                    form.append("V")

        return "".join(form)

    # returns value of M from the [C](VC)^m[V] form
    def get_m(self, word):

        form = self.__get_porter_form(word)
        return form.count("VC")

    # returns true if the stem ends CVC, where the second C is not W, X or Y
    def CVC(self, word):

        return bool(
            (
                self.is_consonant(word, -3)
                and not self.is_consonant(word, -2)
                and self.is_consonant(word, -1)
                and word[-1] not in "wxy"
            )
        )


if __name__ == "__main__":
    x = StemmerUtility()
    word = "elephant"
    print("Input word :", word)
    print(
        "Is the 3rd letter of input word a consonant : ",
        x.is_consonant(word, 3),
    )
    print("Does input word contain any vowels : ", x.contains_vowel(word))
    print("What is the M score of the word in [C](VC)^m[V] form : ", x.get_m(word))
    print(
        "Does the word end in CVC form, where the second C is not W,X or Y : ",
        x.CVC(word),
    )
