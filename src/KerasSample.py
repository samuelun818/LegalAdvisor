import keras
from keras import layers

import numpy as np
import random
import io

def data_prepare():
    path = keras.utils.get_file(
        "nietzsche.txt",
        origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt",
    )
    print(path)
    with io.open(path, encoding="utf-8") as f:
        text = f.read().lower()

    text = text.replace("\n", "")
    print("Corpus length:", len(text))

    chars = sorted(list(set(text)))
    print(chars)

    char_indices = dict((c, i) for i, c in enumerate(chars))
    print(char_indices)
    indices_char = dict((i, c) for i, c in enumerate(chars))
    print(indices_char)
    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 40
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print("Number of sequences:", len(sentences))
    print("sentences:", sentences[0:20])

    x = np.zeros((len(sentences), maxlen, len(chars)), dtype="bool")
    y = np.zeros((len(sentences), len(chars)), dtype="bool")
    print(x.shape, y.shape)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    return x, y, maxlen, chars, char_indices, indices_char, text

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def build(model, text, x,y, maxlen, chars, char_indices, indices_char):
    epochs = 40
    batch_size = 128

    for epoch in range(39, epochs):
        model.fit(x, y, batch_size=batch_size, epochs=epoch)
        print()
        print("Generating text after epoch: %d" % epoch)

        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print("...Diversity:", diversity)

            generated = ""
            sentence = text[start_index: start_index + maxlen]
            print('...Generating with seed: "' + sentence + '"')

            for i in range(400):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.0
                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]
                sentence = sentence[1:] + next_char
                generated += next_char

            print("...Generated: ", generated)
            print("-")


def main():
    X, Y, maxlen, chars, char_indices, indices_char, text = data_prepare()
    print(X[1][0])
    print(Y[1])
    print (len(chars))

    model = keras.Sequential(
        [
            keras.Input(shape=(maxlen, len(chars))),
            layers.LSTM(128),
            layers.Dense(len(chars), activation="softmax"),
        ]
    )
    optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)

    # build(model, text, X,Y, maxlen, chars, char_indices, indices_char)
    return

if __name__ == "__main__":
    main()