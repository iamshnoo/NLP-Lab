# NLP Lab Assignment 1

***Student Details:***

- Name : Anjishnu Mukherjee
- Registration Number : B05-511017020
- Exam Roll Number : 510517086
- Email : 511017020.anjishnu@students.iiests.ac.in

---

***Input files :***

1. ```sample-text-1.txt``` [English]
2. ```sample-text-2.txt``` [Bengali]

***Bengali stopwords file :*** - ```stopwords-bn.txt```

***Output files :***

1. ```en-output.txt``` [English]
2. ```bn-output.txt``` [Bengali]

***ipython notebook*** - ```NLP_Lab_Assignment_1.ipynb```

***Code :***

```run.py```

***Prerequisites :***

```nltk```

---

The code for the assignment is present in run.py, which can be executed using
```python run.py``` from the terminal. It assumes as input the 2 input files and
the bengali stopwords file and produces the 2 output files.

The ipython notebook contains the same code executed interactively within the
jupyter notebook so that the output can be seen at the same time as the code.

Pre-requisite installations include the nltk package and also it is needed to
install nltk_data.

---

***Assignment description :***

Write programs for the following (Use input the English language text and unicoded text for other languages; sample input text files are copied from newspaper and are attached herewith.):

1. Create a small text file, and write a program to read it and print it with a line number at the start of each line. (Make sure you don’t introduce an extra blank line between each line).

2. Define a function called vocab_size(text) that has a single parameter for the text, and which returns the vocabulary size of the text.

3. Write a function named word_freq() that takes a word as input and compute the frequency of the occurrence of the word in that section of the corpus. Test your result with the help of a frequency distribution library function(FreqDist ()) in NLTK.

4. Define a function percent(word, text) that calculates how often a given word occurs in a text and expresses the result as a percentage.

5. Write a function that finds the 10 most frequently occurring words of a text that are not stopwords, contractions or conjunction.

6. Find all the four-letter words from the given text file. Show these words in decreasing order of frequency.

7. Write a program to find all words that occur at least three times in the
   Brown Corpus.

---
