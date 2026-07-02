import argparse
import os
import numpy as np

import re

from matplotlib import pyplot

from Helpers import dataset_helper, log_helper
from Trainers import text_classifier
from Trainers.Category_trainer import  category_trainer
from Crawlers import HKJudgement

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
    print(pred)

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

def transform_judgements(trainer):
    X, Y = [], []
    categorised_dict = {}

    judgements = HKJudgement.HKJudgement()

    path = f"../{judgements.JUDGMENT_PREFIX}/{judgements.LOCATION}/"
    files = os.listdir(path)
    files.sort()

    for filename in files:
        jnum = filename[:-5]
        text = judgements.readJudgement(jnum)
        if text != None:
            text = text.lower()
            words = text.split()
            category = re.sub(r'[^a-zA-Z]', '', words[0])
            if category == "":
                continue

            words = words[1:]
            x = transform_text(trainer, words)
            if x != None:
                X.append(x)
                Y.append(category)

            if category not in categorised_dict:
                categorised_dict.setdefault(category, [])

            categorised_dict[category].extend([text])
            dataset_helper.save_dataset(categorised_dict[category], category)

    print(categorised_dict[Y[0]])
    Y = trainer.encode_category(Y)
    return np.array(X), np.array(Y)

def transform_text(trainer, words):
    article = trainer.encode_article(words)
    return article

def main(arg):
    action = arg['action']
    input = arg['text']

    trainer = category_trainer()
    if action == "train":
        X, Y = transform_judgements(trainer)
        trainer.model_train(X, Y)

        print("Testing loss : {loss}; accuracy: {acc}".format(loss=trainer.loss, acc=trainer.acc))
        return

    if action == "predict":
        if input == "":
            print("Incorrect input.")
            return

        text = input.lower()
        words = text.split()
        x = transform_text(words)

        pred_class = trainer.predict(x)
        print(pred_class)
        return


    #　print("Evaluating Size: {0} / {1}".format(X_test.shape, y_test.shape))
    # trainer.result_evaluate(X_test, y_test)


    # if action == "train":
    #     train_model()
    # elif action == "predict":
    #     predict_result(input)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str,  help="Action (train/predict)")
    parser.add_argument("text", type=str, nargs='?')
    args = parser.parse_args()
    args = vars(args)

    main(args)