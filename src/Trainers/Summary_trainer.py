import re
import numpy as np

import keras
from keras import Sequential
from keras.layers import Dense, LSTM, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

from matplotlib import pyplot

from .BagOfWord import bag_of_words

class summary_trainer:
    def __init__(self):
        self.bag = None
        self.model = None
        self.acc, self.loss, self.history = None, None, None

        self.MODEL_FILE = "./Outputs/Summarisation_model.keras"
        self.PLOTS_DIRECTORY = "../plots/"

        return

    def encode_articles(self, text, no_of_words=10):
        x, y = None, None
        if text is None or text.strip() == "":
            return x, y

        cleaned_text = " ".join(text.splitlines())
        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)
        words = cleaned_text.split()

        if self.bag == None:
            self.bag = bag_of_words(1)
            self.bag.load_wordbag_withstopwords()

        if len(words) <= no_of_words * 2:
            return x, y

        x, y = [], []
        word_x = np.zeros((no_of_words, len(self.bag.words)), dtype="bool")
        word_y = np.zeros((len(self.bag.words)), dtype="bool")
        for i in range(len(words)):

            counter = 0
            next = 0
            while counter < no_of_words:
                if i + counter + next > len(words) - (no_of_words * 2):
                    break

                word = (words[i + counter + next])
                if word in self.bag.words:
                    indices = np.where(self.bag.words == word)

                    word_x[counter][indices[0]] = True
                    next = 0
                else:
                    next = next + 1
                    continue

                if counter == (no_of_words - 1):
                    out_word = (words[i + counter + next + 1])
                    indices = np.where(self.bag.words == out_word)

                    word_y[indices[0]] = True

                counter = counter + 1

            if len(word_x) == no_of_words:
                x.append(word_x)
                y.append(word_y)

        return x, y

    def train_model(self, X, Y):

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
        print("Training Size: {}".format(X_train.shape[0]))

        n_features = X_train.shape[1]
        n_class = y_train.shape[1]
        n_nodes = round(n_class / 10)

        print("Input features: {input}; Output classes: {output}; Training nodes: {nodes}".format(input=n_features,
                                                                                                  output=n_class,
                                                                                                  nodes=n_nodes))

        self.model = Sequential(
            [
                Input(shape=(n_features, n_class)),
                # Bidirectional(LSTM(n_nodes, return_sequences=True)),
                LSTM(1024, recurrent_dropout=0.2, return_sequences=True),
                # Dropout(0.2),
                LSTM(1024, recurrent_dropout=0.2, return_sequences=True),
                # Dropout(0.2),
                LSTM(1024, recurrent_dropout=0.2),
                Dense(n_class, activation="softmax"),
            ]
        )

        optimizer = keras.optimizers.RMSprop(learning_rate=0.05)
        print("Set model")

        plot_model(self.model, to_file='{0}lstm_neural_network.png'.format(self.PLOTS_DIRECTORY),
                   show_shapes=True, show_layer_names=True)


        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        print("Compile model")

        if self.__fit(X_train, y_train, 40):
            self.loss, self.acc = self.model.evaluate(X_test, y_test, verbose=2)
            print("loss: {0}; Accuracy: {1}".format(self.loss, self.acc))

        return

    def __fit(self, X_train, y_train, epochs=40, validation_split=0.2):
        isFitted = False

        if self.model == None:
            print("Invalid model.")
            return isFitted

        self.history = self.model.fit(X_train, y_train, epochs=epochs,  # batch_size=batch,
                                      verbose=2, validation_split=validation_split)

        self.model.save("{0}".format(self.MODEL_FILE))
        isFitted = True

        print("fitted model")
        self.model.summary()
        self.plot_history()

        return isFitted

    def predict_model(self):
        return

    def plot_history(self):
        if self.history is None:
            return

        filename = "{0}SummarizationTraining_history.png".format(self.PLOTS_DIRECTORY)
        # plot loss during training
        pyplot.subplot(211)
        pyplot.title('Loss / Categorical Cross Entropy ({})'.format("LSTM(2048)"))
        pyplot.plot(self.history.history['loss'], label='train')
        pyplot.plot(self.history.history['val_loss'], label='test')
        # plot accuracy during training
        pyplot.subplot(212)
        pyplot.title('Accuracy')
        pyplot.plot(self.history.history['accuracy'], label='train')
        pyplot.plot(self.history.history['val_accuracy'], label='test')
        pyplot.legend()
        pyplot.savefig(filename)