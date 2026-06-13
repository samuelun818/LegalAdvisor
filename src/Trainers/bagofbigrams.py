import sys
sys.path.append("../..")

import numpy as np
import re

import nltk
from nltk import tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from src.Helpers import dataset_helper


class bag_of_bigrams:
    def __init__(self):
        self.grams = []
        self.gramstype = "bigrams"

        nltk.download('punkt_tab')
        nltk.download('words')
        nltk.download('wordnet')

        self.english_words = set(nltk.corpus.words.words())

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
        # try:
        #     return detect(word) == 'en'
        # except LangDetectException:
        #     return False

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

        if len(self.grams) <= 3:
            return

        pre_dataset = dataset_helper.load_dataset("{0}.npy".format(name + self.gramstype))
        self.grams = dataset_helper.merge_dataset(self.grams, pre_dataset)

        dataset_helper.save_dataset(self.grams, name + self.gramstype)

    def init(self):
        self.grams = []

    def __set_bigrams(self, words):
        bigrams = []

        previous_word = ""
        for i in range(len(words)):
            word = ""
            if i < len(words):
                word = words[i]
                word = re.sub(r'[^a-zA-Z]', '', word.lower())

                if word.strip() == "":
                    continue

                if not self.__is_english(word):
                    continue

            bigram = [previous_word, word]
            previous_word = word

            if self.grams.count(bigram) <= 0:
                bigrams.append(bigram)

        return bigrams

    def fill_bag(self, text):
        sentences = tokenize.sent_tokenize(text)

        for sentence in sentences:
            if sentence.strip() == "":
                continue

            words = sentence.split()
            if len(words) < 2 :
                continue
            # print(words)

            sent_bigrams = self.__set_bigrams(words)
            if len(sent_bigrams) < 3 :
                continue

            self.grams.extend(sent_bigrams)
        return

    def __set_bigrams_output(self, sentences):
        if len(sentences) < 2:
            return None, None

        next_sent = sentences[1]

        previous_word = ""
        word = ""
        words = sentences[0].split()
        if len(words) <= 1:
            return None, None

        sent_bigrams = []
        sent_outputs = []
        for i in range(len(words) + 1):
            output_word = ""

            if i < len(words):
                output_word = words[i]
                output_word = re.sub(r'[^a-zA-Z]', '', output_word.lower())

                if output_word.strip() == "":
                    continue

                if not self.__is_english(output_word):
                    continue

            sent_bigrams.append([previous_word, word])
            sent_outputs.append(output_word)
            previous_word = word
            word = output_word

        return sent_bigrams, sent_outputs

    def transform_to_grams(self, text):
        text = text.replace("\n", "")
        sentences = tokenize.sent_tokenize(text)
        article = []
        output = []

        for j in range(len(sentences)):
            if sentences[j].strip() == "":
                continue

            sent_bigrams, sent_out = self.__set_bigrams_output(sentences[j:j+2])

            if sent_bigrams is not None:
                article.extend(sent_bigrams)
                output.extend(sent_out)

        return article, output

    def regress_to_words(self, grams):
        article = []

        i = 0
        for gram in self.grams:
            i = i + 1
            if gram > 0:
                article.append(self.grams[i][0])

        return article