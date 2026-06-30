from sklearn import preprocessing
import numpy as np
from enum import Enum

import re
import string

import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from Helpers import dataset_helper

class BagType(Enum):
    Uni = 1
    Bi = 2
    Tri = 3


class bag_of_words:
    def __init__(self, no_of_grams = 1):
        self.words = None
        self.bagtype = "words"
        self.no_of_grams = no_of_grams
        self.le = preprocessing.LabelEncoder()

        nltk.download('stopwords')
        nltk.download('punkt_tab')
        nltk.download('words')
        nltk.download('wordnet')

        self.english_words = set(nltk.corpus.words.words())
        self.stop_words = []
        for s in stopwords.words('english'):
            self.stop_words.append(re.sub(r'[^a-zA-Z]', '', s))

        return

    def Is_english(self, words):
        isallenglish = True
        for word in words:
            word = word.lower()
            if not isallenglish:
                break

            if not word in self.english_words:
                n = WordNetLemmatizer().lemmatize(word, pos='n')

                if not n in self.english_words:
                    v = WordNetLemmatizer().lemmatize(word, pos='v')

                    if not v in self.english_words:
                        aj = WordNetLemmatizer().lemmatize(word, pos='a')

                        if not aj in self.english_words:
                            av = WordNetLemmatizer().lemmatize(word, pos='r')

                            if not av in self.english_words:
                                isallenglish = False

        return isallenglish

    def fill_wordbag(self, paragraph):
        sentences = tokenize.sent_tokenize(paragraph)

        self.words = []
        self.words.append("")

        for sentence in sentences:
            if sentence.strip() == "":
                continue

            words = sentence.split()
            if (len(words)) < self.no_of_grams - 1:
                continue

            sent_grams = []
            if (self.no_of_grams == 1):
                sent_grams = self.set_unigram(words)
            else:
                sent_grams = self.set_multigram(words)

            if len(sent_grams) > 0:
                for gram in sent_grams:
                    if self.words.count(gram) <= 0:
                        self.words.append(gram)
        return

    def set_unigram(self, words):
        grams = []
        if len(words) < self.no_of_grams:
            return []

        for word in words:
            if not self.Is_english(word):
                continue

            if word in self.stop_words:
                continue

            if grams.count(word) <= 0:
                grams.append(word)

        return grams

    def set_multigram(self, words):
        return

    def save_wordbag(self):
        filename = "bagof{}".format(self.bagtype)
        dataset_helper.save_dataset(np.array(self.words), filename)

    def load_wordbag(self):
        filename = "bagof{}.npy".format(self.bagtype)
        self.words = dataset_helper.load_dataset( filename)

    def merge_wordbag(self):
        filename = "bagof{}".format(self.bagtype)

        pre_dataset = dataset_helper.load_dataset("{}.npy".format(filename))
        if (pre_dataset is not None):
            if (len(np.array(self.words).shape) != len(pre_dataset.shape)):
                return

        self.words = dataset_helper.merge_dataset(self.words, pre_dataset)

        dataset_helper.save_dataset(self.words, filename)

    def init_wordbag(self):
        self.words = None

    def shape(self):
        return np.array(self.words).shape

    def vectorize_text(self, text):


        return # X, y