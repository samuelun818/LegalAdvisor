import numpy as np

from keras import Sequential
from tensorflow.keras.layers import Dense

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from .bagofunigrams import bag_of_unigrams

CLASS_IDS = [
    "Aeroplane",
    "Bicycle",
    "Bird",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cow",
    "Dining Table",
    "Dog",
    "Horse",
    "Motorbike",
    "Person",
    "Potted Plant",
    "Sheep",
    "Sofa",
    "Train",
    "Tvmonitor",
    "Total",
]

class text_classifier:
    def __init__(self, modelfile='JudicalClassification.keras', batchsize=32, epochs=150, act='relu', init='he_normal'):
        self.bagofunigrams = None
        self.resultclass = None

        self.batch_size = batchsize
        self.epochs = epochs
        self.act = act
        self.init = init
        self.model_file= modelfile

        self.labelencoder = LabelEncoder()
        self.model = Sequential()
        self.history = None
        return

    def __encode(self, text):
        bag = bag_of_unigrams.load()
        gram = []

        for word in bag:
            gram.append(text.count(word))

        return gram

    def __decode(self, grams):
        return

    def transform_classes(self, classes):
        y_classes = []
        classes = np.array(classes)
        t_class = self.labelencoder.fit_transform(classes)
        n_yclass = t_class.max()
        self.resultclass = []

        for tclass in t_class:
            yclass = []

            if self.resultclass.count(tclass) <= 0 :
                self.resultclass.append(tclass)

            for i in range(n_yclass + 1):
                if i == tclass:
                    yclass.append(1)
                else:
                    yclass.append(0)
            y_classes.append(yclass)

        return np.array(y_classes)


    def train(self, X, y, train_size=0.3):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_size)

        print(X_train.shape, X_test.shape)
        print(y_train.shape, y_test.shape)

        n_features = X_train.shape[1]
        n_class = y_train.shape[1]

        print("Input features: {input}; Output classes: {output}".format(input=n_features, output=n_class))
        # input layer and first hidden layer, n_features = input_nodes, hidden layer with 10 nodes
        self.model.add(Dense(100, activation=self.act, kernel_initializer=self.init, input_shape=(n_features,)))
        self.model.add(Dense(500, activation=self.act, kernel_initializer=self.init))
        # self.model.add(Dense(1500, activation=self.act, kernel_initializer=self.init))
        self.model.add(Dense(500, activation=self.act, kernel_initializer=self.init))
        # self.model.add(Dense(50, activation=self.act, kernel_initializer=self.init))
        self.model.add(Dense(n_class, activation='softmax'))
        # model.add(Dense(n_class, activation = 'sigmoid'))

        print("Set model")

        # compile the layers to model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("Compiled model")

        # fit the model
        self.history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=self.epochs,
                            batch_size=self.batch_size, verbose=1)
        print("fitted model")

        self.model.save("models/{0}".format(self.model_file) )
        loss, acc = self.model.evaluate(X_test, y_test, verbose=2)
        #
        return loss, acc

    def predict(self, X):
        self.model.load("models/{0}".format(self.model_file))
        x_gram = self.__encode(X)

        pred_Y = None
        pred_Y = self.model.predict(x_gram)
        print(pred_Y)
        return pred_Y

