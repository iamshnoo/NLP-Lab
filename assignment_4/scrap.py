import requests
from bs4 import BeautifulSoup
from contextlib import redirect_stdout

from constants import blog_urls, test_urls


class WebScraper:
    def __init__(self, urls: list) -> None:
        self.urls = urls

    def __process_single_url(self, url: str):
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html5lib")
        article = soup.find("div", attrs={"class": "post-content"})
        return article.text

    def __generate_corpus(self):
        self.data = ""
        for url in self.urls:
            self.data += self.__process_single_url(url)

    def save_corpus(self, filename: str) -> None:
        self.__generate_corpus()
        if filename:
            with open(filename, "w") as data_file:
                with redirect_stdout(data_file):
                    print(self.data)


if __name__ == "__main__":

    model = WebScraper(blog_urls)
    model.save_corpus("training_data.txt")

    scraper = WebScraper(test_urls)
    scraper.save_corpus("testing_data.txt")
