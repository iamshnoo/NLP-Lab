import random
from collections import namedtuple
from typing import Optional

import pandas as pd
from nltk.tokenize import sent_tokenize

from utils import cfg


class Dataset(
    namedtuple(
        "_Dataset",
        "movie_names",
    )
):
    def __new__(
        cls,
        seed: Optional[int] = cfg["SEED"],
        num_movies: Optional[int] = cfg["NUM_MOVIES"],
        num_sentences: Optional[int] = cfg["NUM_SENTENCES"],
        wikipedia_movies: Optional[str] = cfg["WIKIPEDIA_MOVIES_DATASET_KAGGLE"],
        data_file: Optional[str] = cfg["DATA_FILE"],
    ):
        # split data into training and testing set
        if seed is not None:
            random.seed(seed)

        # load the dataset -> https://www.kaggle.com/jrobischon/wikipedia-movie-plots
        df = pd.read_csv(wikipedia_movies)
        # shuffle the dataframe randomly and extract the names of all movies
        movies = df["Title"].sample(frac=1, random_state=seed).to_list()

        with open(data_file, "w") as f:
            counter = 1
            movie_names = []
            # choose a movie if length og its plot has more than 10 sentences
            for movie in movies:
                if counter <= num_movies:
                    movie_plot = df[df["Title"] == movie].iloc[0]["Plot"].rstrip("\r\n")
                    sentences = sent_tokenize(movie_plot)
                    # randomly sample n=10 sentences from the movie plot
                    if len(sentences) >= num_sentences:
                        sentence_sample = random.sample(sentences, num_sentences)
                        for sentence in sentence_sample:
                            f.write(sentence + " ")
                        f.write("\n")
                        counter += 1
                        movie_names.append(movie)

        return super().__new__(cls, movie_names)

    def __len__(self):
        return len(self.movie_names)


if __name__ == "__main__":
    data = Dataset()
    print("-" * 120)
    print("The movies chosen for this dataset are : ")
    print("-" * 120)
    print(data.movie_names)
    print("-" * 120)
