import numpy as np
import os

import helpers.file_helper as fh

DATAFILE_FOLDER = "datasets/"

def load_dataset(file_name):

    if DATAFILE_FOLDER not in file_name:
        file_name = DATAFILE_FOLDER + file_name;

    if not fh.exists(file_name):
        return None

    dataset = np.load(file_name , allow_pickle=True)

    return dataset

def save_dataset(dataset, file_name):

    if DATAFILE_FOLDER not in file_name:
        file_name = DATAFILE_FOLDER + file_name

    np.save(file_name, dataset)

def remove_dataset(file_name):

    if DATAFILE_FOLDER not in file_name:
        file_name = DATAFILE_FOLDER + file_name;

    fh.delete(file_name)

def extend_dataset(data, target):

    data = np.array(data)
    target = np.array(target)

    if target is None or target.shape[0] == 0:
        target = data
    else:
        target = np.vstack((target, data))
        target = np.array(target)

    return target


def append_dataset(data, target):
    data = np.array(data)

    if target is None or len(target) <= 0:
        target = data
    else:
        target = np.concatenate((target, data), axis=0)

    return target

def merge_dataset(data, target, sort=True):
    target = append_dataset(data, target)
    target = np.unique(target,  axis=0)

    return target

def save_arrays(x, y, file_name):
    if DATAFILE_FOLDER not in file_name:
        file_name = DATAFILE_FOLDER + file_name

    np.savez(file_name, x=x, y=y)