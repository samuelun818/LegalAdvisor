import os
import time
import argparse
import requests

import numpy as np

from Trainers import bagofbigrams, bagofunigrams, bagoftrigrams
from Helpers import log_helper, html_helper, file_helper, dataset_helper

# url = 'https://legalref.judiciary.hk/lrs/common/ju/ju_body.jsp?DIS=167483&AH=&QS=&FN=&currpage=T#'
HK_JUDGMENT_URL = 'https://legalref.judiciary.hk/lrs/common/ju/ju_body.jsp'
HK_JUDGMENT_HEADER = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
JUDGMENT_PATH = "judgments/"


def download_judgments(judgment_number):
    judgment_file = JUDGMENT_PATH + str(judgment_number) + ".html"

    if os.path.isfile(judgment_file):
        log_helper.print_message('Already exist judgment: ' + str(judgment_number))
        return

    if judgment_number % 100 == 0:
        time.sleep(2)

    url = HK_JUDGMENT_URL + '?DIS=' + str(judgment_number) + '&AH=&QS=&FN=&currpage=T#'
    response = requests.get(url, headers=HK_JUDGMENT_HEADER)

    if response.status_code == 200:
        html_helper.writ_html(judgment_file, response.content)

        log_helper.print_message('Download judgment: ' + str(judgment_number))

def read_judgment(judgment_number):
    judgment_file = JUDGMENT_PATH + str(judgment_number) + ".html"

    if not file_helper.exists(judgment_file):
        log_helper.print_message('extract_judgment: Judgment is not exist : ' + str(judgment_number))
        return

    text = html_helper.read_html(judgment_file)
    if text.strip() == "":
        log_helper.print_message('extract_judgment: Judgment content is not valid : ' + str(judgment_number))
        return None

    return text

def extract_judgements(bagofgrams, start_num, end_num):
    bagofgrams.delete("judgements")

    for num in range(start_num, end_num):
        text = read_judgment(num)
        if text is None or text.strip() == "":
            continue

        bagofgrams.fill_bag(text)
        log_helper.print_message('Extract judgment: {0} ({1})'.format(str(num), str(bagofgrams.shape())), True)

        bagofgrams.merge("judgements")
        if num >= end_num - 1:
            log_helper.print_message('Save bag of words: {0}'.format(str(bagofgrams.shape())))

        bagofgrams.init()

def transform_judgements(bagofgrams, action, start_num, end_num):
    bagofgrams.load("judgements")
    dataset_helper.remove_dataset("judgments_{0}.npz".format(action))

    log_helper.print_message('Load bag of words: {0}'.format(str(bagofgrams.shape())))
    for num in range(start_num, end_num):
        text = read_judgment(num)
        if text is None or text.strip() == "":
            continue

        article, category = bagofgrams.transform_to_grams(text)  # transform_grams(text, np.array(bag_of_grams))
        log_helper.print_message('Transform judgment content: {0} {1}'.format(str(num), str(np.array(article).shape)), False)

        if article is None or len(article) <= 0:
            continue

        merge_judgements(action, num, article, category)

def merge_judgements(action, num, article, category):
    pre_articles, pre_categories = [], []
    pre_articleset = dataset_helper.load_dataset("judgments_{0}.npz".format(action))

    if pre_articleset != None:
        pre_articles = pre_articleset['x']
        pre_categories = pre_articleset['y']

    articles = dataset_helper.extend_dataset(article, pre_articles)
    categories = dataset_helper.append_dataset(category, pre_categories)

    articles = np.array(articles)
    categories = np.array(categories)
    print(articles.shape, categories.shape)

    dataset_helper.save_arrays(articles, categories, 'judgments_{}'.format(action))
    log_helper.print_message('Save judgment content: {0} {1}'.format(str(num), str(articles.shape)),
                             True)


def main(arg):
    action = arg['action']
    start_num = arg['start']
    end_num = arg['end']

    log_helper.print_message('Action: {0}'.format(action))
    if action == "download":
        for num in range(start_num, end_num):
            download_judgments(num)

        log_helper.print_message("Download completed.")
        return

    bagofgrams = None
    if action == "unigrams":
        bagofgrams = bagofunigrams.bag_of_unigrams()
    elif action == "bigrams":
        bagofgrams = bagofbigrams.bag_of_bigrams()
    elif action == "trigrams":
        bagofgrams = bagoftrigrams.bag_of_trigrams()
    # elif action =="multigrams":
    #     bagofgrams = bagoftrigrams.bag_of_trigrams()
    else:
        return

    log_helper.print_message('Bag type : {0}'.format(bagofgrams.type()))
    extract_judgements(bagofgrams, start_num, end_num)

    log_helper.print_message('Fit Bag of grams : {0}'.format(bagofgrams.type()))
    transform_judgements(bagofgrams, action, start_num, end_num)
    #
    # log_helper.print_message("Article grams : " + str(np.array(articles).shape))
    # log_helper.print_message("Output categories : " + str(np.array(category).shape))
    # dataset_helper.save_arrays(np.array(articles), np.array(category), 'judgments_{}'.format(action))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str)
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    args = parser.parse_args()
    args = vars(args)

    main(args)
