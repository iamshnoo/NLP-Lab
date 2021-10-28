# NLP Lab Assignment 6

***Student Details:***

- Name : Anjishnu Mukherjee
- Registration Number : B05-511017020
- Exam Roll Number : 510517086
- Email : 511017020.anjishnu@students.iiests.ac.in

---

***Input files :***

1. ```demo_grammar.txt```
2. ```pcfg.txt```
3. ```nltk_test.txt```
4. ```test_sentences.txt```

***Output files :***

1. ```q1.txt```
2. ```test_parse_trees```

***Code :***

1. ```dataset.py``
2. ```cky.py```

***Prerequisites :***

1. ```tqdm```
2. ```nltk```
3. ```sklearn```
4. ```re```

---

Run ```dataset.py``` to create ```pcfg.txt``` containing grammar rules in
CNF form extracted from a portion of NLTK Penn Treebank dataset and some test
sentences from some testing trees chosen from the same dataset.
```DEMO``` is a flag on line 7 of ```cky.py```.
Run ```cky.py``` with ```DEMO = True``` to run on the demo grammar with a single
test sentence, and ```DEMO = False``` to run on the grammar extracted from
Treebank for all the sentences in ```test_sentences.txt```. Running with
```DEMO = False``` takes a long time (a few minutes atleast). In the driver
method, ```CKY.print_tree(write=True, draw=False)``` is called to produce only
written output. If ```CKY.print_tree(write=True, draw=True)``` is called, then
the syntax tree for each sentence, as parsed using probabilistic CKY algorithm
and the input ```pcfg.txt```, is drawn on a canvas window using the
```.draw()``` method of nltk. Note that this drawing won't work on google colab,
as far as I know. All the canvases drawn using this method are screenshotted and
stored in the folder ```test_parse_trees``` for reference.

---

***Assignment description :***

1. Write production rules for checking the syntax of the sentences of the following subcategories of English language:
 -  Sentence with declarative structure.
 -  Sentences with imperative structure
 -  Sentences with yes-no question structure
 -  Sentences with various wh-structures (who, whose, when, where, what, which, how, why).

2.  Design/Implement a Probabilistic CKY parser for the same mentioned above. Use an available Treebank (if any) or Create your own hand-checked parsed Training set and test set.
     Test your parser by parsing the test set and compute the labeled recall and labeled precision.

3. Associate the following Feature structures: a)  agreement for Number and Person.  Check the compatibility of the input test sentence.

---
