"""this script performs hyperparamter tunning and saves logs for tensorboard visualizations"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorboard.plugins.hparams import api as hp
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/data_processed2")
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 256
data["target"] = data["bucket"].astype('category').cat.codes
data.drop("bucket", axis = 1, inplace = True)
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
    """cast features to tf.float"""
    return tf.cast(features, tf.float32), label

train_ds = train_ds.map(normalize_ds, num_parallel_calls = AUTOTUNE)
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(buffer_size=len(train_ds))
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(AUTOTUNE)

test_ds = test_ds.map(normalize_ds, num_parallel_calls = AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(AUTOTUNE)


def train_model_num_epochs(hparams, num_epochs):
    """train model for num_epochs and save logs both for training and test required for tensorboard"""
    train_step = 0
    units = hparams[HP_NUM_UNITS]
    drop_rate = hparams[HP_DROPOUT]
    learning_rate = hparams[HP_LR]
    activation = hparams[HP_ACTIVATION]
    optimizer = keras.optimizers.Adam(learning_rate = learning_rate)

    model = keras.Sequential(
        [
                keras.Input(shape = (2)),
                layers.Dense(units, activation = activation),
                layers.Dense(64, activation = activation),
                layers.Dense(32, activation = activation),
                layers.Dropout(drop_rate),
                layers.Dense(850)
            ]

    )
    #write to TB
    run_dir = (
        "logs/train/"
        +str(units)
        +"units_"
        + str(drop_rate)
        + "dropout_"
        +str(learning_rate)
        +"learning_rate_"
        +str(activation)
        +"_activation_fn"
    )

    for epoch in range(num_epochs):
        #backprop and collecting train loss/accuracy
        for batch_idx, (x,y) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                y_pred = model(x, training = True)
                loss = loss_fn(y,y_pred)

            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            acc_metric.update_state(y, y_pred)

        #collecting test accuracy
        for batch_idx, (x,y) in enumerate(test_ds):
            with tf.GradientTape() as tape:
                y_pred_test = model(x, training = False)
                loss_test = loss_fn(y, y_pred_test)

            acc_metric_test.update_state(y, y_pred_test)

        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)
            tf.summary.scalar("loss", loss, step=train_step)
            accuracy = acc_metric.result()
            tf.summary.scalar("accuracy", accuracy, step = train_step)
            accuracy_test = acc_metric_test.result()
            tf.summary.scalar("accuracy_test", accuracy_test, step = train_step)
            train_step += 1

        acc_metric.reset_states()
        acc_metric_test.reset_states()

#loss function
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits = True)

#metrics
acc_metric = keras.metrics.SparseCategoricalAccuracy()
acc_metric_test = keras.metrics.SparseCategoricalAccuracy()

#hparam grid
HP_NUM_UNITS = hp.HParam("num units", hp.Discrete([128,256,512]))
HP_DROPOUT = hp.HParam("dropout", hp.Discrete([0.1,0.2]))
HP_LR = hp.HParam("learning_rate", hp.Discrete([1e-1,1e-2, 1e-3]))
HP_ACTIVATION = hp.HParam("activation", hp.Discrete(["relu","linear"]))

for lr in HP_LR.domain.values:
    for units in HP_NUM_UNITS.domain.values:
        for rate in HP_DROPOUT.domain.values:
            for activation in HP_ACTIVATION.domain.values:
                hparams = {
                    HP_LR : lr,
                    HP_NUM_UNITS : units,
                    HP_DROPOUT : rate,
                    HP_ACTIVATION: activation
                }
                train_model_num_epochs(hparams, num_epochs = 300)