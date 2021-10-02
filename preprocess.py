import pandas as pd
import numpy as np
from functions import one_hot_encoding
import os

def preprocess(data):

    # put all coordinate values into [0,1] range
    data["xphase3"] = data["xphase3"] / 816000
    data["yphase3"] = data["yphase3"] / 816000

    # convert x,y coordinates from phase4 to 200m x 200m buckets
    custom_bucket_array = np.linspace(0, 816000, 41)
    data["cutx"] = pd.cut(data["xphase4"], custom_bucket_array)
    data["cuty"] = pd.cut(data["yphase4"], custom_bucket_array)
    data["bucket"] = data["cutx"].astype(str) + data["cuty"].astype(str)
    data.drop(data.columns[0], axis=1, inplace=True)
    data.drop(["yphase4", "xphase4", "cutx", "cuty"], axis=1, inplace=True)

    # do one hot encoding for buckets
    data = one_hot_encoding(data)
    data.drop("bucket", axis=1, inplace=True)

    return data


def save_to_csv(data, path):
    """save pandas dataframe as csv file"""
    data.to_csv(path, index=False)


data = pd.read_csv("data/training_dataset_full")
#preprocess to buckets because regression/single point prediction is skewed towards the middle of map
data = preprocess(data)
save_to_csv(data, os.getcwd() + "\data\data_processed")