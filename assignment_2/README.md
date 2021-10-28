# NLP Lab Assignment 2

***Student Details:***

- Name : Anjishnu Mukherjee
- Registration Number : B05-511017020
- Exam Roll Number : 510517086
- Email : 511017020.anjishnu@students.iiests.ac.in

---

***Input files :***

1. ```shakespeare.txt```
2. ```big.txt```

***Output files :***

1. ```med-output.txt```
2. ```autocorrect-output.txt```

***ipython notebook*** - ```NLP_Lab_Assignment_2.ipynb```

***Code :***

1. ```med.py```
2. ```autocorrect.py```

---

The code for Minimum Edit Distance is present in med.py (```python med.py```)
The code for AutoCorrect is present in autocorrect.py (```python autocorrect.py```)
The autocorrect.py file assumes as input, either big.txt or shakespeare.txt, or
any other english text file, for creating word corpuses that can be used for
some probability calculations.
Both the code files need to be present in the same folder, as the autocorrect
file imports the med module from the edit distance file.
The output files for each are provided as examples.

The ipython notebook contains the same code executed interactively,
so that the output can be seen at the same time as the code.

There are no pre-requisite installations needed.

---

***Assignment description :***

Write Programs for the following:

1. Given two strings, the source string X of length n, and target string Y of length m, define D[i, j] as the edit distance between X[1 . . . i] and Y[1 . . . j], i.e., the first i characters of X and the first j characters of Y. The edit distance between X and Y is thus D[n, m]. Write a program to compute the edit distance, D[n, m] between X and Y using the MED algorithm as discussed in the class.

2. Choose a different path through the backpointers and reconstruct its alignment. How many different optimal alignments are there? Show all. Use your hand-computed results to check your code.

4. Create a simple autocorrect algorithm using minimum edit distance and dynamic programming.

---
