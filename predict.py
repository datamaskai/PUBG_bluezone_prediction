import tensorflow as tf
from tensorflow import keras
import pandas as pd
import re
import matplotlib.pyplot as plt
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#load model
model = keras.models.load_model("saved_model/epoch_750")

#load data because we need to connect integer targets to coordinates
data = pd.read_csv("data/data_processed2")
data["target"] = data["bucket"].astype('category').cat.codes
data = data[["target", "bucket"]]

# make dictionary connecting integers and area buckets
my_dict = dict(zip(data["target"], data["bucket"]))

def get_elements_and_positions(lst, n):
    """get elements and their positions of n largest in a list""" "maybe here we need to move position for one"
    return sorted(zip(lst, range(len(lst))), reverse=True)[:n]

def create_prediction_dataframe(x, y, number_of_candidates):
    mydf = pd.DataFrame(get_elements_and_positions(model.predict(tf.constant([[x, y]]))[0], number_of_candidates))
    mydf.columns = ["chance", "target"]
    mydf["area"] = mydf["target"].map(my_dict)
    return mydf,x,y

#helper functions to get coordinates from "area"
def extract_decimal_numbers(string):
    return re.findall(r'\d+', string)

def remove_zeros(numbers):
    return [x for x in numbers if x != '0']

def convert_numbers(list):
    return [int(i) for i in list]

def plot_prediction(mydf, x, y):

    img = plt.imread("./support_image/Miramar_remaster_map.jpg")
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(img, extent=[0, 816000, 816000, 0])
    ax.scatter(x * 816000, y * 816000, marker = "x", s=200, color = "red")

    for i in range(len(mydf)):
        edges = convert_numbers(remove_zeros(extract_decimal_numbers(mydf.iloc[i]["area"])))
        chances = mydf.iloc[i]["chance"]

        ax.plot([edges[0], edges[1], edges[1], edges[0], edges[0]], [edges[2], edges[2], edges[3], edges[3], edges[2]],
                "b")
        ax.fill([edges[0], edges[1], edges[1], edges[0], edges[0]], [edges[2], edges[2], edges[3], edges[3], edges[2]],
                "w", alpha=chances)

        ax.text(edges[0] + 1000, (edges[3]+edges[2])/2 + 1000, round(chances*100,0).astype(int), fontsize = 9, color = "black")

    circle1 = plt.Circle((x*816000, y*816000),65900, color ="w", fill = False, linewidth= 2)
    ax.add_artist(circle1)

    candidate_circle = mydf.loc[mydf['chance'].idxmax()]
    edges_candidate = convert_numbers(remove_zeros(extract_decimal_numbers(candidate_circle["area"])))
    circle2 = plt.Circle(((edges_candidate[0]+edges_candidate[1])/2, (edges_candidate[2]+edges_candidate[3])/2), 36200, color = "w", fill = False,linewidth=2)
    ax.add_artist(circle2)

    plt.show()


def predict(x,y,number_of_candidates):

    #make predictions
    mydf, x, y = create_prediction_dataframe(x,y,number_of_candidates)

    #plot predictions
    plot_prediction(mydf, x, y)




if __name__ == "__main__":
  x = float(sys.argv[1])
  y = float(sys.argv[2])
  number_of_candidates = int(sys.argv[3])
  predict(x,y,number_of_candidates)


