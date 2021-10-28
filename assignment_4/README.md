# NLP Lab Assignment 4

***Student Details:***

- Name : Anjishnu Mukherjee
- Registration Number : B05-511017020
- Exam Roll Number : 510517086
- Email : 511017020.anjishnu@students.iiests.ac.in

---

***Code :*** -

1. ```scrap.py``` - scrapes blogs and stores the text in them
2. ```process.py``` - process scraped data to form a corpus
3. ```utils.py``` - utility functions for text processing
4. ```constants.py``` - urls to be scraped for training and testing data, and
   delimiter tokens
5. ```unigram.py``` - all ngram functionalities for n=1
6. ```bigram.py``` - all ngram functionalities for n=2
7. ```trigram.py``` - all ngram functionalities for n=3
8. ```ngrams.py``` - general ngram class with all functionalities
9. ```demo.py```  - demonstrate the working of the good turing smoothing algorithm
10. ```nltk_test.py``` - simplistic test to check the performance of nltk
    modules on same training data

***Folder Structure :***

```bash
assignment_4
├── README.md
├── bigram.py
├── constants.py
├── demo.py
├── demo_data.txt
├── ngrams.py
├── nltk_test.py
├── outputs
│   ├── bigram
│   │   ├── 2gram-add-one.txt
│   │   ├── 2gram.txt
│   │   ├── bigram-add-one.txt
│   │   ├── good_turing_smoothing_bigram.txt
│   │   └── good_turing_smoothing_bigram_result.txt
│   ├── ngram
│   │   ├── good_turing_smoothing_ngram.txt
│   │   ├── good_turing_smoothing_ngram_result.txt
│   │   ├── ngram-add-one.txt
│   │   ├── ngram.txt
│   │   └── ngram_perplexity.txt
│   ├── nltk_perplexity.txt
│   ├── trigram
│   │   ├── 3gram-add-one.txt
│   │   ├── 3gram.txt
│   │   ├── good_turing_smoothing_trigram.txt
│   │   └── good_turing_smoothing_trigram_result.txt
│   └── unigram
│       ├── 1gram-add-one.txt
│       ├── 1gram.txt
│       ├── good_turing_smoothing_unigram.txt
│       └── good_turing_smoothing_unigram_result.txt
├── process.py
├── processed_data.txt
├── scrap.py
├── testing_data.txt
├── training_data.txt
├── trigram.py
├── unigram.py
└── utils.py
```

***Prerequisites :***

1. NLTK
2. NLTK data
3. Python 3.6+

```bash
pip install requests
pip install html5lib
pip install bs4
```

---

***Assignment description :***

The assignments on Probabilistic Language Model are as follows:

1. Write a program to compute unsmoothed unigrams, bigrams and Trigrams for a sequence of words. Generalize the programs for n-grams.

2. Create a corpus by collecting 50 web news, Create a list of frequency of words and then build a Unigram, bigram and trigram language models, respectively.

3. Use at least three sentences as input to test your language models with any two smoothing techniques. Also, make some performance analysis to explain which model and smoothing method is better.

4. Test your result with the help of the library functions available in NLTK.

---
