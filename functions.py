import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import re


def one_hot_encoding(df):
    """
       :param df: dataframe
       :return: dataframe with one hot encoding
       """
    df = pd.concat([df, pd.get_dummies(df['bucket'])], axis=1)
    return df


def df_to_dataset(df, shuffle=True, batch_size=16):
    """ A utility method to create a tf.data dataset from a Pandas Dataframe"""

    df = df.copy()
    target = df.iloc[:, 2:]

    df = df.iloc[:, :2]
    # df.pop("bucket")

    ds = tf.data.Dataset.from_tensor_slices((dict(df), target.values))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds


def extract_decimal_numbers(string):
    return re.findall(r'\d+', string)

def remove_zeros(numbers):
    return [x for x in numbers if x != '0']

def convert_numbers(list):
    return [int(i) for i in list]


def make_prediction(x, y, model, train, number_of_zones=10,):
    mydf = train.iloc[0:1]
    mydf["xphase3"] = x
    mydf["yphase3"] = y

    test_ds = df_to_dataset(mydf, shuffle=False, batch_size=1)

    prediction = pd.DataFrame(model.predict(test_ds))

    prediction.columns = train.columns[2:]

    # percentages
    percentages = prediction.T.sort_values(by=0, ascending=False).values[:number_of_zones]

    # boxes
    boxes = prediction.T.sort_values(by=0, ascending=False).index[:number_of_zones]

    return x, y, percentages, boxes


def plot_prediction(percentages, boxes, x, y):
    img = plt.imread("./support_image/Miramar_remaster_map.jpg")
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(img, extent=[0, 816000, 816000, 0])
    ax.scatter(x * 816000, y * 816000, s=100)

    # we use counter for shading boxes
    counter = 0

    for box in boxes:
        edges = convert_numbers(remove_zeros(extract_decimal_numbers(box)))

        ax.plot([edges[0], edges[1], edges[1], edges[0], edges[0]], [edges[2], edges[2], edges[3], edges[3], edges[2]],
                "r-")
        ax.fill([edges[0], edges[1], edges[1], edges[0], edges[0]], [edges[2], edges[2], edges[3], edges[3], edges[2]],
                "g", alpha=percentages[counter][0] * 2)

        counter += 1

    plt.show()






