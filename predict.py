import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from functions import make_prediction
from functions import plot_prediction
import pandas as pd
from sklearn.model_selection import train_test_split
import sys


def predict(x,y,number_of_zones):

  #define feature layer
  feature_columns = []

  #numeric cols
  feature_columns.append(tf.feature_column.numeric_column("xphase3"))
  feature_columns.append(tf.feature_column.numeric_column("yphase3"))

  feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


  #initialize model and load saved weights
  model = tf.keras.Sequential([
    feature_layer,
    tf.keras.layers.Dense(128, activation='linear'),
    tf.keras.layers.Dropout(0.10),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(850, activation = "softmax")
  ])

  model.load_weights("./checkpoints/my_checkpoint")

  #load data
  data = pd.read_csv("data/data_processed")
  #divide data into train,val,test
  train, val = train_test_split(data, test_size = 0.1)
  val, test = train_test_split(val, test_size = 0.3)

  x,y,percentages,boxes = make_prediction(x,y, model = model, train = train, number_of_zones=number_of_zones)

  plot_prediction(percentages, boxes, x,y)


if __name__ == "__main__":
  x = float(sys.argv[1])
  y = float(sys.argv[2])
  number_of_zones = int(sys.argv[3])
  predict(x,y,number_of_zones)

