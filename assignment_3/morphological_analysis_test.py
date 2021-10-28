import unittest
from contextlib import redirect_stdout
from morphological_analysis import MorphologicalAnalyzer
from constants import morphology_tests_dict


class TestMorphologicalAnalyzer(unittest.TestCase):
    def test_morphological_analysis_outputs(self) -> None:
        analyzer = MorphologicalAnalyzer()

        for i, (key, value) in enumerate(morphology_tests_dict.items(), start=1):
            result = analyzer.analyze(key)
            self.assertListEqual(result, value)
            print("-" * 120)
            print("Test Case", i, ", Input Word :", key)
            print("Output Morphologies :", result)
            if i == 8:
                print("-" * 120)


if __name__ == "__main__":
    with open("morphological_analysis-output.txt", "w") as f1:
        with redirect_stdout(f1):
            unittest.main()
