# NLP Lab Assignment 7

***Student Details:***

- Name : Anjishnu Mukherjee
- Registration Number : B05-511017020
- Exam Roll Number : 510517086
- Email : 511017020.anjishnu@students.iiests.ac.in

---

***Input files :***

```wiki_movie_plots_deduped.csv``` -> [Kaggle link](https://www.kaggle.com/jrobischon/wikipedia-movie-plots)

***Output files :***

1. ```movies.txt```
2. ```inverted_index.txt```
3. ```term_document_matrix.csv```

***Code :***

1. ```utils.py```
2. ```data.py```
3. ```ir.py```

***Prerequisites :***

1. ```pandas```
2. ```nltk```

---

Run ```data.py``` to create ```movies.txt```.
Run ```ir.py``` to create ```inverted_index.txt```,
```term_document_matrix.csv``` and output a demo run of all functionality.

---

***Assignment description :***

In this assignment use movie's descriptions (a short overview) as the test collection. Create a file (movies.txt) which will contain the description of 50 movies. This file will contain a collection of movie's descriptions with 10 sentences per line per movie (or per document). You can treat each line as a document in this exercise. Preprocess (Tokenization, Removal of stopwords and perform stemming) the documents and write the program for the following.

Create an Inverted Index file
                       a) The first task is to convert the data into an inverted index. You can extract the terms together with document IDs corresponding to the line number. This list should be sorted
                       b) Get the 10 most prominent terms (terms with the largest document frequency)

Term - Document Matrix
                  Write a program that reads through the list of sorted terms and creates a Term - Document Matrix.

Boolean Queries
                      a)  Now test some simple boolean queries with the indexed data / Term - Document Matrix
                      b)  Test conjunctive boolean queries consisting of One or
                      multi-word query terms.

---
