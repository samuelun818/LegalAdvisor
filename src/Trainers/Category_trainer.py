import numpy as np
import re
import pickle

from .BagOfWord import bag_of_words

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras import Sequential
from keras.layers import Dense, Input, Dropout

class category_trainer:
    def __init__(self):
        self.bag = None
        self.model = None
        self.history, self.loss, self.accuracy = None, None, None
        self.result = None
        self.modelfile = "./Outputs/Classification_model.keras"
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
        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(self.le, f)

        y = np.zeros((len(categories), category_y.max()), dtype="bool")
        for i in range(len(category_y)):
            y[i][category_y[i]-1] = True

        return y

    def decode_category(self, codes):
        with open("label_encoder.pkl", "rb") as f:
            self.le = pickle.load(f)

        categories = self.le.inverse_transform(codes)
        return categories

    def model_train(self, X, Y):

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
        print("Training Size: {}".format(X_train.shape[0]))

        n_features = X_train.shape[1]
        n_class = y_train.shape[1]

        self.model = Sequential()
        print("Input features: {input}; Output classes: {output}".format(input=n_features, output=n_class))
        # input layer and first hidden layer, n_features = input_nodes, hidden layer with 10 nodes
        self.model.add(Input(shape=(n_features,)))
        # self.model.add(Dense(16384, activation="tanh", kernel_initializer='he_normal'))
        # self.model.add(Dense(16384, activation="tanh", kernel_initializer='he_normal'))
        # self.model.add(Dropout(0.3))
        # self.model.add(Dense(4096, activation="tanh", kernel_initializer='he_normal'))
        # self.model.add(Dense(4096, activation="tanh", kernel_initializer='he_normal'))
        # self.model.add(Dropout(0.3))
        self.model.add(Dense(2048, activation="tanh", kernel_initializer='he_normal'))
        self.model.add(Dense(2048, activation="tanh", kernel_initializer='he_normal'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1024, activation="tanh", kernel_initializer='he_normal'))
        self.model.add(Dense(1024, activation="tanh", kernel_initializer='he_normal'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(512, activation="tanh", kernel_initializer='he_normal'))
        self.model.add(Dense(512, activation="tanh", kernel_initializer='he_normal'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(256, activation="tanh", kernel_initializer='he_normal'))
        self.model.add(Dense(256, activation="tanh", kernel_initializer='he_normal'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(n_class, activation='sigmoid'))

        print("Set model")

        # compile the layers to model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("Compiled model")

        # fit the model
        self.history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                      epochs=40, verbose=1)
        print("fitted model")

        self.model.save("{0}".format(self.modelfile))
        self.loss, self.acc = self.model.evaluate(X_test, y_test, verbose=2)
        #
        self.model.summary()
        print("loss: {0}; Accuracy: {1}".format(self.loss, self.acc))

    def model_predict(self, X):
        pred = []

        if self.model == None:
            self.model.load("{0}".format(self.modelfile))

        pred = self.model.predict(X)
        pred = self.decode_category(pred)

        return pred