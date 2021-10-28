from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# CONFIGURATION
cfg = {
    "SEED": 42,  # seed for random
    "NUM_MOVIES": 50,  # number of movies for preparing dataset
    "NUM_SENTENCES": 10,  # number of lines to be chosen from each movie plot
    "WIKIPEDIA_MOVIES_DATASET_KAGGLE": "wiki_movie_plots_deduped.csv",  # original data source of 35,000 movies
    "DATA_FILE": "movies.txt",  # 50 movies, 10 lines of plot each, 1 line for each movie
    "INVERTED_INDEX_FILE": "inverted_index.txt",  # inverted index created from movies.txt (OrderedDict)
    "TERM_DOCUMENT_MATRIX_FILE": "term_document_matrix.csv",  # document matrix created from movies.txt (Dataframe)
}

cfg["num_documents"] = cfg["NUM_MOVIES"]

stopwords_list = stopwords.words("english")
stemmer = PorterStemmer()
ignored = string.punctuation + "£" + "–" + "—" + "0123456789"


def process_sentence(sentence):
    # tokenize a sentence
    y = word_tokenize(sentence)
    # remove punctuations, numbers and other special symbols and words
    # containing any of these
    words = [word.lower() for word in y if all(char not in ignored for char in word)]
    # remove the stopwords and stem the remaining words using porter stemmer
    return [stemmer.stem(word) for word in words if word not in stopwords_list]
