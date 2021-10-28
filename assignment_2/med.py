import functools
import math
from contextlib import redirect_stdout
from typing import Dict, List, Optional


# utility function
def ignore_unhashable(func):
    """[To allow caching the distance matrix in its original form, which is a
        list of lists. Reference : ->
        https://stackoverflow.com/a/64111268/11009359]

    Args:
        func ([type]): [The function where we intend to apply this decorator.]
    """
    uncached = func.__wrapped__
    attributes = functools.WRAPPER_ASSIGNMENTS + ("cache_info", "cache_clear")

    @functools.wraps(func, assigned=attributes)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError as error:
            if "unhashable type" in str(error):
                return uncached(*args, **kwargs)
            raise

    wrapper.__uncached__ = uncached
    return wrapper


class MinimumEditDistance:
    def __init__(
        self,
        insertion_cost: Optional[int] = 1,
        deletion_cost: Optional[int] = 1,
        substitution_cost: Optional[int] = 2,
    ) -> None:
        """[Creates an instance for the class.]

        Args:
            insertion_cost (Optional[int], optional): [description]. Defaults to 1.
            deletion_cost (Optional[int], optional): [description]. Defaults to 1.
            substitution_cost (Optional[int], optional): [description]. Defaults to 2.
        """
        self.__insertion_cost = insertion_cost
        self.__deletion_cost = deletion_cost
        self.__substitution_cost = substitution_cost

    def align(self, query: str, target: str) -> Dict:
        """[Compute edit distance. Return a dictionary containing the edit
            distance, the matrix from Wagner Fischer algorithm, a single optimal
            alignment, and a list of all possible alignments.]

        Args:
            query (str): [The query string.]
            target (str): [The target string.]

        Returns:
            Dict: [Wagner Fischer results, single optimal alignment, all optimal
            alignments]
        """
        wagner_fischer_results = self.weightedWagnerFischer(query, target)

        single_optimal_alignment = self.getOptimalAlignment(
            query, target, wagner_fischer_results["distance_matrix"]
        )

        all_optimal_alignments = self.getAllOptimalAlignments(
            query, target, wagner_fischer_results["distance_matrix"]
        )

        return {
            "wagner_fischer_results": wagner_fischer_results,
            "single_optimal_alignment": single_optimal_alignment,
            "all_optimal_alignments": all_optimal_alignments,
        }

    def niceMatrix(self, matrix: List) -> None:
        """[Prints a matrix in a nice human readable format.]

        Args:
            matrix (List): [The matrix which will be printed, stored as a list
            of lists.]
        """
        print("\n".join("".join("{:4}".format(item) for item in row) for row in matrix))

    def niceAlignment(self, alignment: Dict) -> None:
        """[Prints the alignment of query and target in a nice human readable
        format.]

        Args:
            alignment (Dict): [A dictionary containing the aligned query and
            target strings, and a string for the actions needed for alignment.
            For the string of operations, d is for delete, i is for insertion, |
            is for match, s is for substitution. For the aligned target and
            query strings, - corresponds to an insertion or deletion.]
        """
        query, target, operations = (
            alignment["aligned_query"],
            alignment["aligned_target"],
            alignment["operations"],
        )
        print(" ".join(query) + "\n" + " ".join(operations) + "\n" + " ".join(target))

    def weightedWagnerFischer(self, query: str, target: str) -> Dict:
        """[Implements the Wagner Fischer algorithm. Details of the algorithm
        can be found here ->
        https://en.wikipedia.org/wiki/Wagner–Fischer_algorithm .]

            Args:
                query (str): [The query string.]
                target (str): [The target string.]

            Returns:
                Dict: [Contains the edit distance between query and target
                strings and also the distance matrix.]
        """
        len1, len2 = len(query), len(target)
        distance = [[0 for i in range(len2 + 1)] for j in range(len1 + 1)]

        # query prefixes can be transformed into empty target by deleting all
        # characters
        for i in range(len1 + 1):
            distance[i][0] = i * self.__deletion_cost

        # target prefixes can be reached from empty query prefix by inserting
        # every character
        for i in range(len2 + 1):
            distance[0][i] = i * self.__insertion_cost

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                diagonal = distance[i - 1][j - 1] + (
                    0 if query[i - 1] == target[j - 1] else self.__substitution_cost
                )
                left = distance[i][j - 1] + self.__insertion_cost
                up = distance[i - 1][j] + self.__deletion_cost

                distance[i][j] = min(diagonal, left, up)

        edit_distance = distance[len1][len2]
        return {"edit_distance": edit_distance, "distance_matrix": distance}

    @staticmethod
    def reverseString(input_string: str) -> str:
        """[Utility method to reverse a string.]

        Args:
            input_string (str): [The string to be reversed.]

        Returns:
            str: [The reversed string.]
        """
        start = stop = None
        step = -1
        reverse_slice = slice(start, stop, step)
        return input_string[reverse_slice]

    def getOptimalAlignment(self, query: str, target: str, distance: List) -> Dict:
        """[Get a single optimal alignment from the edit distance matrix.]

        Args:
            query (str): [The query string.]
            target (str): [The target string.]
            distance (List): [The edit distance matrix from Wagner Fischer
            algorithm.]

        Returns:
            Dict: [A dictionary containing the aligned query and target strings
            and a string of the operations needed for tha alignment.]
        """
        aligned_query, aligned_target, operation = "", "", ""
        i, j = len(query), len(target)

        while i != 0 or j != 0:

            # find the value of diagonal, top and left cells if they exist
            is_match = False
            diagonal, top, left = math.inf, math.inf, math.inf
            if i != 0 and j != 0:
                is_match = query[i - 1] == target[j - 1]
                diagonal = distance[i - 1][j - 1] + (
                    0 if is_match else self.__substitution_cost
                )
            if i != 0:
                top = distance[i - 1][j] + self.__deletion_cost
            if j != 0:
                left = distance[i][j - 1] + self.__insertion_cost

            # find global min of the three cells
            min_dist = min(diagonal, top, left)

            # check which of those 3 cells correspond to the global min
            if diagonal == min_dist:
                aligned_query += query[i - 1]
                aligned_target += target[j - 1]
                operation += "|" if is_match else "s"
                i -= 1
                j -= 1
            elif top == min_dist:
                aligned_query += query[i - 1]
                aligned_target += "-"
                operation += "d"
                i -= 1
            else:
                aligned_query += "-"
                aligned_target += target[j - 1]
                operation += "i"
                j -= 1

        aligned_query, aligned_target, operation = (
            self.reverseString(aligned_query),
            self.reverseString(aligned_target),
            self.reverseString(operation),
        )

        return {
            "aligned_query": aligned_query,
            "aligned_target": aligned_target,
            "operations": operation,
        }

    def getAllOptimalAlignments(self, query: str, target: str, distance: List) -> List:
        """[Wrapper method that calls a private method which performs
        backtracking to return all optimal alignments. ]

        Args:
            query (str): [The query string.]
            target (str): [The target string.]
            distance (List): [The edit distance matrix from Wagner Fischer.]

        Returns:
            List: [A list of dictionaries, where each dictionary is an
            alignment.]
        """
        self.__alignments = []
        self.__get_all_optimal_alignments(
            query, target, distance, len(query), len(target), "", "", ""
        )
        return self.__alignments

    @ignore_unhashable
    @functools.lru_cache(maxsize=10)
    def __get_all_optimal_alignments(
        self,
        query: str,
        target: str,
        distance: List,
        i: int,
        j: int,
        aligned_query: str,
        aligned_target: str,
        operation: str,
    ) -> None:  # sourcery no-metrics
        """[A backtracking algorithm that finds all paths from distance[m][n] to
        distance[0][0], where distance is the edit distance matrix from weighed
        Wagner Fischer algorithm, and m and n are the lengths of the query and
        target strings respectively. The paths found are the represented as
        alignments which are then stored in a private class member list. Up to
        10 most recent calls are cached in a lru cache.]

        Args:
            query (str): [The query string.]
            target (str): [The target string.]
            distance (List): [The edit distance matrix from weighted Wagner
            Fischer algorithm.]
            i (int): [The starting index for this recursive backtracking
            algorithm, for query string.]
            j (int): [The starting index for this recursive backtracking
            algorithm, for target string.]
            aligned_query (str): [Aligned query string.]
            aligned_target (str): [Aligned target string.]
            operation (str): [String of operations for converting query to
            target.]

        """

        if i == 0 and j == 0:
            optimal_alignment = {"aligned_query": self.reverseString(aligned_query)}
            optimal_alignment["aligned_target"] = self.reverseString(aligned_target)
            optimal_alignment["operations"] = self.reverseString(operation)
            self.__alignments.append(optimal_alignment)

        else:

            # find the value of diagonal, top and left cells if they exist
            is_match = False
            diagonal, top, left = math.inf, math.inf, math.inf
            if i != 0 and j != 0:
                is_match = query[i - 1] == target[j - 1]
                diagonal = distance[i - 1][j - 1] + (
                    0 if is_match else self.__substitution_cost
                )
            if i != 0:
                top = distance[i - 1][j] + self.__deletion_cost
            if j != 0:
                left = distance[i][j - 1] + self.__insertion_cost

            # find global min of the three cells
            min_dist = min(diagonal, top, left)

            # check which of those 3 cells correspond to the global min
            if diagonal == min_dist:
                self.__get_all_optimal_alignments(
                    query,
                    target,
                    distance,
                    i - 1,
                    j - 1,
                    aligned_query + query[i - 1],
                    aligned_target + target[j - 1],
                    operation + ("|" if is_match else "s"),
                )
            if top == min_dist:
                self.__get_all_optimal_alignments(
                    query,
                    target,
                    distance,
                    i - 1,
                    j,
                    aligned_query + query[i - 1],
                    aligned_target + "-",
                    operation + "d",
                )
            if left == min_dist:
                self.__get_all_optimal_alignments(
                    query,
                    target,
                    distance,
                    i,
                    j - 1,
                    aligned_query + "-",
                    aligned_target + target[j - 1],
                    operation + "i",
                )


def human_readable_matrix(str1, str2, matrix):
    x = list(str2)
    x.insert(0, "       ")
    x.insert(1, "#")
    y = list(str1)
    y.insert(0, "#")
    for i, row in enumerate(matrix):
        row.insert(0, y[i])
    matrix.insert(0, x)


if __name__ == "__main__":
    str1 = "TELEPHONE"
    str2 = "ELEPHANT"

    med = MinimumEditDistance(1, 1, 2)
    alignment = med.align(str1, str2)

    with open("med-output.txt", "w") as f1:
        with redirect_stdout(f1):
            print()
            print(
                "Edit Distance : ",
                alignment["wagner_fischer_results"]["edit_distance"],
            )
            print("-" * 80)
            print()
            print("Edit Distance Matrix :")
            matrix = alignment["wagner_fischer_results"]["distance_matrix"]
            human_readable_matrix(str1, str2, matrix)
            med.niceMatrix(matrix)
            print("-" * 80)
            print()
            print("Single Optimal Alignment : ")
            med.niceAlignment(alignment["single_optimal_alignment"])
            print("-" * 80)
            print()
            print("All Optimal Alignments : ")
            for i in range(len(alignment["all_optimal_alignments"])):
                optimal_alignment = alignment["all_optimal_alignments"][i]
                print("-" * 30)
                print("Alignment #", i + 1)
                print("-" * 30)
                med.niceAlignment(optimal_alignment)
            print("-" * 80)
            print()
