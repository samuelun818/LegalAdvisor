import sys
sys.path.append("..")

import numpy as np
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk import tokenize
from nltk.stem.wordnet import WordNetLemmatizer


from helpers import dataset_helper


class bag_of_unigrams:
    def __init__(self):
        self.grams = []
        self.gramstype = "unigrams"

        nltk.download('stopwords')
        nltk.download('punkt_tab')
        nltk.download('words')
        nltk.download('wordnet')

        self.english_words = set(nltk.corpus.words.words())

        self.stop_words = []
        for s in stopwords.words('english'):
            self.stop_words.append(re.sub(r'[^a-zA-Z]', '', s))

        return


    def __is_english(self, word):
        word = word.lower()
        if not word in self.english_words:
            n = WordNetLemmatizer().lemmatize(word, pos='n')

            if not n in self.english_words:
                v = WordNetLemmatizer().lemmatize(word, pos='v')

                if not v in self.english_words:
                    aj = WordNetLemmatizer().lemmatize(word, pos='a')

                    if not aj in self.english_words:
                        av = WordNetLemmatizer().lemmatize(word, pos='r')

                        if not av in self.english_words:
                            return False

        return True


    def fill_bag(self, text):
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

        text = text.replace("\n", "")
        text = text.translate(translator)
        words = (" ".join(text.lower().split())).split()

        for word in words:
            if not self.__is_english(word):
                continue

            if word in self.stop_words:
                continue

            if self.grams.count(word) <= 0:
                self.grams.append(word)



    def type(self):
        return self.gramstype

    def shape(self):
        return np.array(self.grams).shape

    def save(self, name=""):
        if name != "":
            name = name + "_"
        dataset_helper.save_dataset(np.array(self.grams), name + self.gramstype)

    def load(self, name=""):
        if name != "":
            name = name + "_"

        self.grams = dataset_helper.load_dataset("{0}.npy".format(name + self.gramstype))

    def delete(self, name=""):
        if name != "":
            name = name + "_"

        dataset_helper.remove_dataset("{0}.npy".format(name + self.gramstype))

    def merge(self, name=""):
        if name != "":
            name = name + "_"

        pre_dataset = dataset_helper.load_dataset("{0}.npy".format(name + self.gramstype))
        self.grams = dataset_helper.merge_dataset(self.grams, pre_dataset)

        dataset_helper.save_dataset(self.grams, name + self.gramstype)

    def append(self, list):
        self.grams = dataset_helper.append_dataset(list, self.grams)

    def init(self):
        self.grams = []

    def list(self):
        return self.grams.tolist()

    def transform_to_grams(self, text):
        article = []
        category = []

        text = text.lower()
        words = text.split()

        category.append( re.sub(r'[^a-zA-Z]', '', words[0]))
        words = words[1:]

        for gram in self.grams:
            article.append(words.count(gram))

        if (article is None or len(article) < 100):
            article = None
            category = None

        return article, category

    def regress_to_words(self, grams):
        article = []

        i = 0
        for gram in grams:
            i = i + 1
            if gram > 0:
                article.append(self.grams[i])

        return