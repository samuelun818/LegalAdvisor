import argparse
import os
import numpy as np
import gc
import re

from matplotlib import pyplot

from Helpers import dataset_helper, log_helper
from Trainers.Category_trainer import  category_trainer
from Crawlers import HKJudgement


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
            x = trainer.encode_article(words)
            if x != None:
                X.append(x)
                Y.append(category)

            if category not in categorised_dict:
                categorised_dict.setdefault(category, [])

            categorised_dict[category].append(jnum)

    dataset_helper.save_dataset(np.array(categorised_dict), "categorised_jnum")

    Y = trainer.encode_category(Y)
    return np.array(X), np.array(Y)

def main(arg):
    action = arg['action']
    input = arg['text']

    trainer = category_trainer()
    if action == "train":
        X, Y = transform_judgements(trainer)
        trainer.train_model(X, Y)

        print("Testing loss : {loss}; accuracy: {acc}".format(loss=trainer.loss, acc=trainer.acc))
        return

    if action == "predict":
        if input == "":
            print("Incorrect input.")
            return

        text = input.lower()
        words = text.split()
        x = []
        x.append(trainer.encode_article(words))

        pred_class = trainer.predict_category(np.array(x))
        print(pred_class)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str,  help="Action (train/predict)")
    parser.add_argument("text", type=str, nargs='?')
    args = parser.parse_args()
    args = vars(args)

    main(args)
    gc.collect()