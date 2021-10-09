import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/data_processed2")
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 256

#encode categories as integers
data["target"] = data["bucket"].astype('category').cat.codes
data.drop("bucket", axis = 1, inplace = True)

#divide into train/test
train, test = train_test_split(data, test_size = 0.2)

def df_to_dataset(df):
    """create tf.data.Dataset object from a dataframe"""
    dataframe = df.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dataframe,labels))
    return ds

train_ds = df_to_dataset(train)
test_ds = df_to_dataset(test)

def normalize_ds(features, label):
    """cast features to tf.float32"""
    return tf.cast(features, tf.float32), label

#create datasets
train_ds = train_ds.map(normalize_ds, num_parallel_calls = AUTOTUNE)
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(buffer_size=len(train_ds))
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(AUTOTUNE)

test_ds = test_ds.map(normalize_ds, num_parallel_calls = AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(AUTOTUNE)


class MyModel(keras.Model):
    def __init__(self, num_classes = 850):
        super(MyModel, self).__init__()
        self.dense1 = layers.Dense(512, activation = "relu")
        self.dense2 = layers.Dense(64, activation = "relu")
        self.dense3 = layers.Dense(32, activation = "relu")
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)
        self.classifier = layers.Dense(num_classes,activation = "softmax")

    def call(self, input_tensor, training = False):
        x = self.dense1(input_tensor, training = training)
        x = self.dropout1(x,training = training)
        x = self.dense2(x,training = training)
        x = self.dense3(x, training = training)
        x = self.dropout2(x, training = training)
        return self.classifier(x)

model = MyModel(num_classes=850)

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = False), #we are using false in order to get probabilities
    optimizer = keras.optimizers.Adam(0.01),
    metrics = ["accuracy"]
)

#callbacks
def scheduler(epoch, lr):
    """learning rate scheduler function, we are slightly decreasing learning rate after a certain number of epochs"""
    if epoch < 300:
        return lr
    else:
        return lr * 0.99
lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler,
                                                     verbose = False)

#train model and save weights every 10 epochs to complete_saved_model/epoch folder

num_epochs = 800
for epoch in range(num_epochs):
    model.fit(train_ds, epochs = 1, verbose = 2, callbacks = [lr_scheduler])

    if epoch < 10:
        model.save(f"saved_model/epoch_{epoch}")
    if epoch < 100 and epoch % 10 == 0:
        model.save(f"saved_model/epoch_{epoch}")
    elif epoch % 50 == 0:
        model.save(f"saved_model/epoch_{epoch}")



model.evaluate(test_ds, batch_size = 32, verbose = 2)

