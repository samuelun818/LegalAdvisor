import numpy as np
import re

import sys
sys.path.append("..")

import nltk
from nltk.corpus import stopwords
from nltk import tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from .bagofgrams import bag_of_grams
import string

class bag_of_trigrams(bag_of_grams):
    def __init__(self):
        super().__init__()
        self.gram_type = "trigrams"
        self.numberofgram = 3
        return

    # def __is_english(self, word):
    #     word = word.lower()
    #     if not word in self.english_words:
    #         n = WordNetLemmatizer().lemmatize(word, pos='n')
    #
    #         if not n in self.english_words:
    #             v = WordNetLemmatizer().lemmatize(word, pos='v')
    #
    #             if not v in self.english_words:
    #                 aj = WordNetLemmatizer().lemmatize(word, pos='a')
    #
    #                 if not aj in self.english_words:
    #                     av = WordNetLemmatizer().lemmatize(word, pos='r')
    #
    #                     if not av in self.english_words:
    #                         return False

        # return True
        # try:
        #     return detect(word) == 'en'
        # except LangDetectException:
        #     return False

    def type(self):
        return self.gram_type

    # def shape(self):
    #     return super(bag_of_trigrams, self).shape()

    def save(self, name):
        if name != "":
            name = name + "_"

        super(bag_of_trigrams, self).save(name + "_" + self.gram_type)
        # dataset_helper.save_dataset(np.array(self.grams), name + "_" + self.gram_type)

    def load(self, name):
        if name != "":
            name = name + "_"

        super(bag_of_trigrams, self).load("{0}.npy".format(name + self.gram_type))
        # dataset_helper.load_dataset("{0}.npy".format(name + self.gramstype)))


    def delete(self, name=""):
        if name != "":
            name = name + "_"

        super(bag_of_trigrams, self).delete("{0}.npy".format(name + self.gram_type))

    def merge(self, name=""):
        if name != "":
            name = name + "_"

        super(bag_of_trigrams, self).merge("{0}.npy".format(name + self.gram_type))

    def __set_trigrams(self, words):
        trigrams = []

        previous_word_1 = ""
        previous_word_2 = ""
        for i in range(len(words)):
            word = ""
            if i < len(words):
                word = words[i]
                word = re.sub(r'[^a-zA-Z]', '', word.lower())

                if word.strip() == "":
                    continue

                if not super(bag_of_trigrams, self).Is_english(word):
                    continue

            trigram = [previous_word_1, previous_word_2, word]
            previous_word_1 = previous_word_2
            previous_word_2 = word

            if self.grams.count(trigram) <= 0:
                trigrams.append(trigram)

        return trigrams

    def fill_bag(self, text):
        sentences = tokenize.sent_tokenize(text)

        for sentence in sentences:
            if sentence.strip() == "":
                continue

            words = sentence.split()
            if len(words) < 2:
                continue
            # print(words)

            sent_trigrams = self.__set_trigrams(words)
            if len(sent_trigrams) < 2:
                continue

            self.grams.extend(sent_trigrams)
        return

    def __set_grams_output(self, words):
        sent_trigrams = []
        sent_outputs = []

        if len(words) < self.numberofgram:
            return None, None

        count = 0
        for i in range(2, len(words)-1):
            grams = [words[j] for j in [i-2, i-1, i, i + 1]]

            if not self.Is_english(grams):
                continue

            if(grams.count("involuntary")):
                print(i, grams)

            count = count + 1

            sent_trigrams.append(grams[:3])
            sent_outputs.append(grams[3])

        return sent_trigrams, sent_outputs

    def transform_to_grams(self, text):
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

        text = text.replace("\n", "")
        text = text.translate(translator)
        words = (" ".join(text.lower().split())).split()

        article, output = self.__set_grams_output(words)
        return article, output

    def regress_to_words(self, grams):
        return
