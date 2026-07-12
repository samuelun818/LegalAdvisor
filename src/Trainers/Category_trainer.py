import numpy as np
import re
import pickle

from .BagOfWord import bag_of_words

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from keras import Sequential
from keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import plot_model

from matplotlib import pyplot

class category_trainer:
    def __init__(self):
        self.bag = None
        self.model = None
        self.history, self.loss, self.accuracy = None, None, None
        self.result = None
        self.MODEL_FILE = "./Outputs/Classification_model.keras"
        self.ENCODER_FILE = "./Outputs/category_encoder.pkl"
        self.PLOTS_DIRECTORY = "../plots/"
        self.le = LabelEncoder()
        self.loss = None
        self.acc = None

    def encode_article(self, words):
        if self.bag == None:
            self.bag = bag_of_words(1)
            self.bag.load_wordbag()

        article_x = []
        for gram in self.bag.words:
            article_x.append(words.count(gram))

        if (len(article_x) < 100):
            article_x = None

        return article_x

    def encode_category(self, categories):
        categories = np.squeeze(categories)
        category_y = self.le.fit_transform(categories)
        with open(self.ENCODER_FILE, "wb") as f:
            pickle.dump(self.le, f)

        y = np.zeros((len(categories), category_y.max()), dtype="bool")
        for i in range(len(category_y)):
            y[i][category_y[i]-1] = True

        return y

    def decode_category(self, codes):
        with open(self.ENCODER_FILE, "rb") as f:
            self.le = pickle.load(f)

        categories = self.le.inverse_transform(codes)
        return categories

    def train_model(self, X, Y):

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        print("Training Size: {}".format(X_train.shape[0]))

        n_features = X_train.shape[1]
        n_class = y_train.shape[1]

        self.model = Sequential()
        print("Input features: {input}; Output classes: {output}".format(input=n_features, output=n_class))
        # input layer and first hidden layer, n_features = input_nodes, hidden layer with 10 nodes
        self.model.add(Input(shape=(n_features,)))
        # self.model.add(Dense(16384, activation="tanh", kernel_initializer='he_normal'))
        # self.model.add(Dropout(0.3))
        # self.model.add(Dense(16384, activation="tanh", kernel_initializer='he_normal'))

        self.model.add(Dense(4096, activation="tanh", kernel_initializer='he_normal'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(4096, activation="tanh", kernel_initializer='he_normal'))

        # self.model.add(Dense(2048, activation="tanh", kernel_initializer='he_normal'))
        # self.model.add(Dense(2048, activation="tanh", kernel_initializer='he_normal'))
        # self.model.add(Dropout(0.3))
        self.model.add(Dense(1024, activation="tanh", kernel_initializer='he_normal'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1024, activation="tanh", kernel_initializer='he_normal'))

        # self.model.add(Dense(512, activation="tanh", kernel_initializer='he_normal'))
        # self.model.add(Dense(512, activation="tanh", kernel_initializer='he_normal'))
        # self.model.add(Dropout(0.3))
        self.model.add(Dense(256, activation="tanh", kernel_initializer='he_normal'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(256, activation="tanh", kernel_initializer='he_normal'))

        self.model.add(Dense(n_class, activation='sigmoid'))

        print("Set model")

        plot_model(self.model, to_file='{0}dense_neural_network.png'.format(self.PLOTS_DIRECTORY),
                            show_shapes=True, show_layer_names=True)

        # compile the layers to model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("Compiled model")

        # fit the model
        if self.__fit(X_train, y_train, 40):
            self.loss, self.acc = self.model.evaluate(X_test, y_test, verbose=2)
            print("loss: {0}; Accuracy: {1}".format(self.loss, self.acc))

    def __fit(self, X_train, y_train, epochs=40, validation_split=0.2):
        isFitted = False

        if self.model == None:
            print("Invalid model.")
            return isFitted

        self.history = self.model.fit(X_train, y_train, epochs=epochs, # batch_size=batch,
                                      verbose=2, validation_split=validation_split)

        self.model.save("{0}".format(self.MODEL_FILE))
        isFitted = True

        print("fitted model")
        self.model.summary()
        self.plot_history()

        return isFitted

    def predict_category(self, X):
        pred = []

        print(np.array(X).shape)
        if self.model == None:
            self.model = Sequential()
            self.model = keras.models.load_model(self.MODEL_FILE)

        pred = self.model.predict(X)
        pos = np.argmax(pred[0])

        pred_y = self.decode_category([pos])

        return pred_y

    def plot_history(self):
        if self.history is None:
            return

        filename = "{0}CategoriseTraining_history.png".format(self.PLOTS_DIRECTORY)
        # plot loss during training
        pyplot.subplot(211)
        pyplot.title('Loss / Categorical Cross Entropy ({})'.format("DENSE"))
        pyplot.plot(self.history.history['loss'], label='train')
        pyplot.plot(self.history.history['val_loss'], label='test')
        # plot accuracy during training
        pyplot.subplot(212)
        pyplot.title('Accuracy')
        pyplot.plot(self.history.history['accuracy'], label='train')
        pyplot.plot(self.history.history['val_accuracy'], label='test')
        pyplot.legend()
        pyplot.savefig(filename)

