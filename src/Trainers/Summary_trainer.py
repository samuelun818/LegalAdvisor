import re

import keras
from keras.layers import Dense, LSTM

from sklearn.model_selection import train_test_split

class summary_trainer:
    def __init__(self):
        self.model = None
        self.acc, self.loss, self.history = None, None, None

        self.MODEL_FILE = "./Outputs/Summarisation_model.keras"
        self.PLOTS_DIRECTORY = "../plots/"

        return

    def encode_articles(self, text):
        x, y = None, None
        if text is None or text.strip() == "":
            return x, y

        cleaned_text = " ".join(text.splitlines())
        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)

        words = cleaned_text.split()
        if len(words) <= 10:
            return x, y

        for i in range(len(words)):
            if i > len(words) - 12:
                break

            x.append(words[i: i+10])
            y.append(words[1+11])

        return

    def train_model(self):
        return

    def __fit(self):
        return

    def predict_model(self):
        return