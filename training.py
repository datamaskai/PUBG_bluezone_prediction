import tensorflow as tf
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from functions import df_to_dataset
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#load preprocessed data
data = pd.read_csv("data/data_processed")

#divide data into train,val,test
train, val = train_test_split(data, test_size = 0.1)
val, test = train_test_split(val, test_size = 0.3)

#make tensorflow datasets
feature_columns = []

#numeric cols
feature_columns.append(tf.feature_column.numeric_column("xphase3"))
feature_columns.append(tf.feature_column.numeric_column("yphase3"))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

train_ds = df_to_dataset(train, batch_size=256)
val_ds = df_to_dataset(val, shuffle=False, batch_size=8)
test_ds = df_to_dataset(test, shuffle=False, batch_size=8)

print(feature_layer)

#compile and train model
model = tf.keras.Sequential([
  feature_layer,
  tf.keras.layers.Dense(128, activation='linear'),
  tf.keras.layers.Dropout(0.10),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(850, activation = "softmax")
])

model.compile(optimizer = "adam",
             loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics = ["accuracy"])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=800)

model.save_weights(os.getcwd() + "/checkpoints/my_checkpoint")