# NLP Lab Assignment 3

***Student Details:***

- Name : Anjishnu Mukherjee
- Registration Number : B05-511017020
- Exam Roll Number : 510517086
- Email : 511017020.anjishnu@students.iiests.ac.in

---

***Input files :***

1. ```english.hfst```  (downloaded from <https://sourceforge.net/projects/hfst/files/resources/morphological-transducers/hfst-english.tar.gz>)
2. ```fst.txt```  (the above .hfst file is converted to text using the ```hfst-fst2txt``` binary from HFST library <https://hfst.github.io/downloads/index.html>)

***Output files :***

1. ```stemmer_test-output.txt```
2. ```morphological_analysis-output.txt```

***Code :*** -

1. ```constants.py```
2. ```utils.py```
3. ```fst_visualize.py```
4. ```morphological_analysis.py```
5. ```morphological_analysis_test.py```
6. ```prefix_removal.py```
7. ```stemmer.py```
8. ```stemmer_test.py```

***Folder Structure :***

```bash
assignment_3
├── README.md
├── constants.py
├── english.hfst
├── fst.txt
├── fst_visualize.py
├── morphological_analysis-output.txt
├── morphological_analysis.py
├── morphological_analysis_test.py
├── prefix_removal.py
├── stemmer.py
├── stemmer_test-output.txt
├── stemmer_test.py
└── utils.py
```

***Prerequisites :***

1. Graphviz -> ```pip install graphviz```
2. NLTK (PorterStemmer)
3. NLTK data(Wordnet, Words)
4. Libhfst -> ```pip install hfst```
5. Python 3.6+

---

All the code files are executable themselves and contain a single demo test case
to explain its input and output format. The test files test a larger sample of
inputs to produce the output files.

The stemmer implemented is a naive implementation of the original Porter Stemmer
algorithm described [here](https://tartarus.org/martin/PorterStemmer/def.txt).
The prefix removal functionality adds a feature to this stemmer, by removing
prefixes from words, if the word after removing prefixes is in a dictionary of
words from NLTK wordnet corpus.

For morphological analysis, I use a pre-defined transducer from HFST, which has
approximately 13 lac transitions, and perform lookups on this transducer to
analyse the structure of words.

I have also provided a code for visualizing the transducer. But, because there
are too many transitions, it is not feasible to run it on the entire FST file.
Instead, it makes more sense to use the visualizer to visualize any particular
parts of the FST one might be interested in. (So, just choose a subset of the
total number of lines of fst.txt as input to the visualiser when you want to use
it.)

The pre-defined transducer we are using is written and compiled using the
default OpenFST backend of the HFST library. For this assignment, I used the
compiled version of the FST directly. Grammatical rules used to define the
lexicon that constructs the FST can be found from
[here](<https://sourceforge.net/projects/hfst/files/resources/morphological-transducers/hfst-english.tar.gz>).

A blog describing how HFST can be used to create something like this can be
found [here](https://ftyers.github.io/2017-КЛ_МКЛ/hfst.html).

A blog describing how transducers in HFST work for morphological analysis
can be found [here](https://fomafst.github.io/morphtut.html).

---

***Assignment description :***

1. Write your own Affix removal algorithms for English words. Remove suffixes and/or prefixes from terms leaving a stem. Find the root/stem word. A simple example of an affix removal stemmer is one that removes the plurals from terms. Analyse the inflectional/derivational morphology on different parts of speech, spelling rules, and so on for the mostly used English words.

2. Write the Complete Documentation for the same.

3. Implement your algorithm.
4. Test your result with the help the stemmer available in NLTK.

---
