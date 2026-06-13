import numpy as np
import argparse

from matplotlib import pyplot

from helpers import dataset_helper
from trainers import text_classifier


def load_articles():
    articles = []
    category = []

    articles_set = dataset_helper.load_dataset("judgments_unigrams.npz")
    articles = articles_set['x']
    category = articles_set['y']

    return articles, category

def predict_result(x):
    classifier = text_classifier.text_classifier()

    pred = classifier.predict(x)

def plot_result_history(history):

    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss / Categorical Cross Entropy')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['categorical_accuracy'], label='train')
    pyplot.plot(history.history['val_categorical_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()

def train_model():
    classifier = text_classifier.text_classifier(epochs=250)

    X, Y = load_articles()
    y = classifier.transform_classes(Y)  # LabelEncoder().fit_transform(Y)

    print(X.shape, y.shape)
    loss, acc = classifier.train(X, y)

    print("Models loss : {loss}; accuracy: {acc}".format(loss=loss, acc=acc))
    plot_result_history(classifier.history)


def main(arg):
    action = arg['action']
    input = arg['text']

    if action == "train":
        train_model()
    elif action == "predict":
        predict_result(input)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str)
    parser.add_argument("text", type=str)
    args = parser.parse_args()
    args = vars(args)

    main(args)