import numpy as np
import re

from BagOfWord import bag_of_words

from sklearn.preprocessing import LabelEncoder
from keras import Sequential
from keras.layers import Dense

class class_trainer:
    def __init__(self):
        self.bag = None
        self.model = None
        self.history, self.loss, self.accuracy = None, None, None
        self.result = None
        self.modelfile = "./Outputs/Classification_model.keras"
        self.le = LabelEncoder()

    def Encode_article(self, text):
        if self.bagofword == None:
            self.bag = bag_of_words(1)
            self.bag.load_wordbag()

        article_x, category = [], []
        text = text.lower()
        words = text.split()

        category.append(re.sub(r'[^a-zA-Z]', '', words[0]))
        words = words[1:]
        for gram in self.bag.words:
            article_x.append(words.count(gram))

        if (article_x is None or len(article_x) < 100):
            article_x = None
            category = None

        return article_x, category

