import libhfst
from typing import List, Optional


class MorphologicalAnalyzer:
    def __init__(self, fst: Optional[str] = "english.hfst") -> None:
        # load the transducer file and read it using libhfst python API
        pre_defined_fst = libhfst.HfstInputStream(fst)
        self.fst = pre_defined_fst.read()

    def analyze(self, word: str) -> List:

        # perform lookup from fst to get list of morphologies
        analysis_result = self.fst.lookup(word)

        # process the lookup output to remove unnecessary items
        results = []
        for result in analysis_result:
            result = result[0]
            processed_result = []
            flag = False
            for letter in result:
                if letter == "[":
                    flag = True
                elif letter == "]":
                    flag = False
                if not flag and letter != "]":
                    processed_result.append(letter)
            result = "".join(processed_result)
            result = result.replace("@_EPSILON_SYMBOL_@", "")
            results.append(result)

        # return processed outputs
        return results


if __name__ == "__main__":
    analyzer = MorphologicalAnalyzer()
    word = "cats"
    print("Input word :", word)
    print("Output Morphologies :", analyzer.analyze(word))
