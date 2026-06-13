import numpy as np
import sys
sys.path.append("../..")

import nltk
from nltk.stem.wordnet import WordNetLemmatizer

from src.helpers import dataset_helper


class bag_of_grams:
    def __init__(self):
        self.grams = []

        nltk.download('punkt_tab')
        nltk.download('words')
        nltk.download('wordnet')

        self.english_words = set(nltk.corpus.words.words())
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
        # try:
        #     return detect(word) == 'en'
        # except LangDetectException:
        #     return False


    def shape(self):
        return np.array(self.grams).shape

    def save(self, name=""):
        if name == "":
            return

        dataset_helper.save_dataset(np.array(self.grams), name)

    def load(self, name=""):
        if name == "":
            return

        self.grams = dataset_helper.load_dataset(name)

    def delete(self, name=""):
        if name == "":
            return

        dataset_helper.remove_dataset(name)

    def merge(self, name=""):
        if name == "":
            return

        pre_dataset = dataset_helper.load_dataset(name)
        if (pre_dataset is not None):
            if (len(np.array(self.grams).shape) != len(pre_dataset.shape)):
                return

        self.grams = dataset_helper.merge_dataset(self.grams, pre_dataset)

        name = name[:-4]
        dataset_helper.save_dataset(self.grams, name)

    def init(self):
        self.grams = []
