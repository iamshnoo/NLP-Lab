from collections import OrderedDict
from typing import Optional, List

import pandas as pd
from nltk.tokenize import sent_tokenize

from utils import process_sentence, cfg


class IR:
    def __init__(self) -> None:
        pass

    def create_inverted_index(
        self,
        data_file: Optional[str] = cfg["DATA_FILE"],
        inverted_index_file: Optional[str] = cfg["INVERTED_INDEX_FILE"],
    ) -> None:

        self.inverted_index = {}
        with open(data_file, "r") as f:
            for doc_id, line in enumerate(f):
                sentences = sent_tokenize(line)
                for sentence in sentences:
                    words = process_sentence(sentence)
                    for stem_word in words:
                        try:
                            self.inverted_index[stem_word].add(doc_id)
                        except KeyError:
                            self.inverted_index[stem_word] = {doc_id}

        # sort the inverted index as per the number of documents for each key in
        # reverse order
        self.sorted_inverted_index = OrderedDict(
            sorted(
                self.inverted_index.items(), key=lambda item: len(item[1]), reverse=True
            )
        )

        with open(inverted_index_file, "w") as f:
            self.__write_inverted_index_file(f)

    def __write_inverted_index_file(self, f):
        f.write("-" * 120)
        f.write("\n")
        f.write("{:>20} | {:>90}".format("Term", "Documents it occurs in"))
        f.write("\n")
        f.write("-" * 120)
        f.write("\n")
        for key, value in self.sorted_inverted_index.items():
            f.write("{:>20} | {:>90}".format(key, str(value)))
            f.write("\n")

    def top_n_words(self, n: Optional[int] = 10) -> List:
        return [
            (item[0], len(item[1]))
            for i, item in enumerate(self.sorted_inverted_index.items())
            if i <= n
        ]

    def create_term_document_matrix(
        self,
        term_document_matrix_file: Optional[str] = cfg["TERM_DOCUMENT_MATRIX_FILE"],
    ) -> None:
        # term document matrix
        # number of cols = number of documents
        # number of rows = number of keys of inverted index
        self.term_document_mat = pd.DataFrame(
            [
                [False for _i in range(cfg["num_documents"])]
                for _j in self.sorted_inverted_index.keys()
            ],
            index=self.sorted_inverted_index.keys(),
            columns=[str(i) for i in range(cfg["num_documents"])],
        )

        for word, doc_ids in self.sorted_inverted_index.items():
            for doc_id in doc_ids:
                self.term_document_mat.loc[word][doc_id] = True

        self.term_document_mat.to_csv(term_document_matrix_file)

    def query_inverted_index(self, query_word: str) -> None:
        print("-" * 120)
        print("Querying Inverted Index : ")
        print("-" * 120)
        try:
            doc_set = self.sorted_inverted_index[query_word.lower()]
        except:
            print("Invalid query : ", query_word)
            print("Documents containing query term : ", {})
            print("-" * 120)
        else:
            print("Query successful for input : ", query_word)
            print("Documents containing query term : ", doc_set)
            print("-" * 120)

    def query_term_document_mat(self, query_word: str, query_doc: str) -> None:
        print("-" * 120)
        print("Querying Term Document Matrix : ")
        print("-" * 120)
        try:
            result = self.term_document_mat.loc[query_word.lower()][int(query_doc)]
        except:
            print("Invalid query : ", query_word, query_doc)
            print("-" * 120)
        else:
            print("Query successful for input : ", query_word, query_doc)
            if result:
                print("Document", query_doc, "contains query term", query_word)
            else:
                print("Document", query_doc, "doesn't contain query term", query_word)
            print("-" * 120)

    def boolean_query_test(self, query: str) -> None:
        # 'query' can be (X and Y and Z and ...), (X or Y or Z or ...),
        # or a combination of `and`, `or` between the terms. Each term can be
        # either a valid word from the inverted index or of the form (not A)
        # where A is a valid word from the inverted index.
        # 'modified_query' stores the modified query for final evaluation
        # and -> & (set intersection)
        # or -> | (set union)
        # not -> set difference(all ids - ids of word after not)
        words = query.split()
        modified_query = ""
        i = 0
        while i < len(words):
            if words[i] == "not":
                docs_containing_word = self.sorted_inverted_index[words[i + 1].lower()]
                docs_not_containing_word = {
                    *[j for j in range(50)]
                } - docs_containing_word
                modified_query += str(eval("docs_not_containing_word")) + " "
                i += 1
            elif words[i] == "and":
                modified_query += "& "
            elif words[i] == "or":
                modified_query += "| "
            else:
                docs_containing_word = self.sorted_inverted_index[words[i].lower()]
                modified_query += str(eval("docs_containing_word")) + " "
            i += 1

        # evaluate the whole modified query
        print("-" * 120)
        print("Evaluating boolean query : ", query)
        print("-" * 120)
        print("Documents satisfying query expression : ", eval(modified_query))
        print("-" * 120)

    def demo(self) -> None:
        self.create_inverted_index()
        print("-" * 120)
        print("Top 10 words and corresponding document frequencies : ")
        print("-" * 120)
        print(self.top_n_words())
        print("-" * 120)
        self.create_term_document_matrix()
        self.query_inverted_index("take")
        self.query_inverted_index("tak")
        self.query_term_document_mat("take", "46")
        self.query_term_document_mat("take", "47")
        self.query_term_document_mat("tak", "-1")
        self.boolean_query_test("not take and one and kill")


if __name__ == "__main__":
    information_retriever = IR()
    information_retriever.demo()
