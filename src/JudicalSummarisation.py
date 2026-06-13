import numpy as np
import argparse
import gc

from trainers import bagofunigrams
from trainers import textconverter
from helpers import dataset_helper, plot_helper, log_helper

import keras
from keras.layers import Dense, LSTM

from sklearn.model_selection import train_test_split


def load_articles(filename, loading_size=1):
    articles = []
    category = []

    if loading_size is None:
        loading_size = 1

    articles_set = dataset_helper.load_dataset("{0}.npz".format(filename))
    articles = articles_set['x']
    category = articles_set['y']
    print(articles.shape, category.shape)
    data_size = articles.shape[0]
    print(data_size)
    size = round(data_size * loading_size)
    log_helper.print_message('Load articles : {0} of {1}'.format(str(size), str(data_size)))

    return articles[:size], category[:size]

def fit_converter():
    bagofwords = bagofunigrams.bag_of_unigrams()
    bagofwords.load("judgements")
    bagofwords.append(bagofwords.stop_words)
    bagofwords.append([''])
    log_helper.print_message('Load bag of words : {0}'.format(str(bagofwords.shape())))

    bag = bagofwords.list()
    c = textconverter.text_converter()
    c.fit(bag)
    log_helper.print_message('Fit text converter')

    return c

def transform_data(converter, X, Y):
    print(X[0], Y[0])
    ycode = converter.encode(Y)
    xcode = []
    print(X.shape)
    x = np.zeros((X.shape[0], len(X[0]), len(converter.bagofcodes)), dtype="bool")
    y = np.zeros((X.shape[0], len(converter.bagofcodes)), dtype="bool")
    print(x.shape)
    for i in range(X.shape[0]):
        log_helper.print_message('Transforming ... {0:.3%}'.format(i / X.shape[0]), newline=False)
        words = X[i]
        codes = converter.encode(words)
        for t, code in enumerate(codes):
            x[i, t, code] = 1
        y[i, ycode[i]] = 1

    print(x[0], y[0])
    return np.array(x), np.array(y)

def reverse_data(converter, data):
    print(data.shape)

    code = np.zeros(data.shape[1], dtype="int")
    print(code.shape)

    for i in range(data.shape[1]):
        d = data[0][i]
        indices = np.where(d >= 1)

        if len(indices) > 0:
            indice = indices[0]
            if len(indice):
                code[i] = indice[0]

    print(code)
    return code

def train_model(x_train, y_train, x_test, y_test, batch_size=32, epochs=150, act='relu', init='he_normal'):
    n_features = x_train.shape[1]
    n_class = y_train.shape[1]

    # model = Sequential()
    model = keras.Sequential(
        [
            keras.Input(shape=(n_features, n_class)),
            LSTM(128,  return_sequences=True),
            LSTM(256),
            Dense(n_class, activation="softmax"),
        ]
    )
    print("Input features: {input}; Output classes: {output}".format(input=n_features, output=n_class))
    # input layer and first hidden layer, n_features = input_nodes, hidden layer with 10 nodes
    # model.add(Dense(32, activation=act, kernel_initializer=init, input_shape=(n_features,)))
    # model.add(Dense(128, activation=act, kernel_initializer=init))
    # model.add(Dense(128, activation=act, kernel_initializer=init))
    # model.add(Dense(128, activation=act, kernel_initializer=init))
    # model.add(Dense(256, activation=act, kernel_initializer=init))
    # model.add(Dense(n_class, activation='softmax'))
    # model.add(Dense(n_class, activation = 'sigmoid'))
    # model.add(Dense(n_class))

    print("Set model")

    # compile the layers to model
    loss_fn = keras.losses.CategoricalCrossentropy(
        from_logits=False,       # bool
    )
    optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Compiled model")

    # optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
    # model.compile(loss="categorical_crossentropy", optimizer=optimizer)

    # converted_x = []
    # for i in range(x_train.shape[0]):
    #     code_x = []
    #     for j in range(x_train[i].shape[0]):
    #         code_x.append([True if x == 1.0 else False for x in x_train[i][j]])
    #     converted_x.append(code_x)
    #
    # converted_x = np.array(converted_x)
    # print(converted_x.shape)
    #
    # converted_y = []
    # for i in range(y_train.shape[0]):
    #     converted_y.append([True if y == 1.0 else False for y in y_train[i]])
    #
    # converted_y = np.array(converted_y)
    # print(converted_y.shape)
    #
    # # fit the model
    # print(converted_x[0], converted_y[0])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=0.1)
    print("fitted model")

    model.save("models\JudicalSummarization.keras")
    gc.collect()
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    #
    return loss, acc, history

def predict_result(x):
    pred = []
    model = keras.models.load_model("models\JudicalSummarization.keras")
    model.summary()

    pred = dataset_helper.append_dataset(x, pred)
    print(pred.shape)
    x = np.array(x)
    x_t = x

    for i in range(40):
        y_pred = model.predict(x_t)
        y = y_pred[0]
        # converted_y = []
        print(np.argmax(y), max(y))
        print(sum(num != 0 for num in y))
        converted_y = np.zeros((1, pred.shape[-1]), dtype="bool")
        pred_max = np.argmax(y)
        converted_y[0][pred_max] = True

        print(np.array(converted_y).shape)
        x_t = np.append([x_t[0][1:]], [converted_y], axis=1)
        print(x_t.shape)
        pred = np.append(pred, [converted_y], axis=1)

    print(pred.shape)
    return pred



def main(args):

    action = args['action']

    converter = fit_converter()
    print(converter.encode(["it"]))
    print(converter.decode([3413]))
    gc.collect()

    if action=="transform":
        X, Y = load_articles("judgments_trigrams")
        log_helper.print_message('Loaded articles : X:{0} Y:{1}'.format(str(X.shape), str(Y.shape)))

        X, Y = transform_data(converter, X, Y)

        log_helper.print_message('Transformed X:{0} and Y:{1}'.format(str(X.shape), str(Y.shape)))
        dataset_helper.save_arrays(np.array(X), np.array(Y), 'data_set')

    elif action=="train":
        X, Y = load_articles("data_set", loading_size=args['size'])

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

        loss, acc, hist = train_model(X_train, y_train, X_test, y_test, batch_size=16, epochs=20)
        log_helper.print_message("Models loss : {loss}; accuracy: {acc}".format(loss=loss, acc=acc))

        plot_helper.plot_traininghistory(hist, loss_type="")

    elif action=="predict":
        x = [["divorce", "relation", "claim"]]
        y = ["ground"]

        x, y = transform_data(converter, np.array(x), np.array(y))
        gc.collect()
        # x = x.reshape(x.shape[1], x.shape[2])
        print(x.shape)
        y_pred = predict_result(x)
        gc.collect()

        y_pred = reverse_data(converter, y_pred)
        y_words = converter.decode(y_pred)
        print(y_words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str)
    parser.add_argument("--size", type=float, required=False)
    args = parser.parse_args()
    args = vars(args)

    main(args)
    gc.collect()