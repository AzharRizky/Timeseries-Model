# -*- coding: utf-8 -*-
"""Submission 2 - Time Series.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nKB5oZjJMpi5CwaDgLG96YlJDZw-SrSM

## Azhar Rizki Zulma

Dataset: https://www.kaggle.com/srinuti/residential-power-usage-3years-data-timeseries

**Model Using StartDate & Value (kWh) colum**

### Import Library
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.backend import clear_session

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

"""### Read Data"""

df = pd.read_csv('/power_usage_2016_to_2020.csv')
df

df.info()

df['date']=pd.to_datetime(df['StartDate'])
df.head()

"""Create new dataframe with only 2 colum (date & value)"""

energy=df[['date','Value (kWh)']].copy()
energy['val'] = energy['Value (kWh)']
energy = energy.drop('Value (kWh)', axis=1)
energy.set_index('date', inplace= True)
energy.head()

"""### Timeseries Plot"""

plt.figure(figsize=(20,8))
plt.plot(energy)
plt.title('Timeseries Of Total Energy Using')
plt.xlabel('Datetime')
plt.ylabel('Energy')
plt.show()

"""### Modelling"""

date = df['date'].values
val = energy['val'].values

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1:]))
    return ds.batch(batch_size).prefetch(1)

x_train, x_test, y_train, y_test = train_test_split(val, date, test_size = 0.2, random_state = 0 , shuffle=False)
print(len(x_train), len(x_test))

data_x_train = windowed_dataset(x_train, window_size=60, batch_size=100, shuffle_buffer=5000)
data_x_test = windowed_dataset(x_test, window_size=60, batch_size=100, shuffle_buffer=5000)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="causal", activation="relu", input_shape=[None, 1]),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.LSTM(64, return_sequences=True),
  tf.keras.layers.Dense(30, activation="relu"),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 400)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])

"""**Min & Max Value**"""

min = energy['val'].min()
print('Min Value : ')
print(min)

max = energy['val'].max()
print('Max value : ' )
print(max)

x = (max - min) * (10/100)
print(x)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if((logs.get('mae')< x) and (logs.get('val_mae')< x)):
      self.model.stop_training = True
      print("\nMAE of the model < 10% of data scale")
callbacks = myCallback()

tf.keras.backend.set_floatx('float64')
history = model.fit(data_x_train, epochs=500, validation_data=data_x_test, callbacks=[callbacks])

"""### Plot Loss & Mae"""

loss = history.history['loss']
val_loss = history.history['val_loss']
mae = history.history['mae']
val_mae = history.history['val_mae']

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(loss, label='Training set')
plt.plot(val_loss, label='Test set', linestyle='--')
plt.legend()
plt.grid(linestyle='--', linewidth=1, alpha=0.5)

plt.subplot(1, 2, 2)
plt.title('MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.plot(mae, label='Training set')
plt.plot(val_mae, label='Test set', linestyle='--')
plt.legend()
plt.grid(linestyle='--', linewidth=1, alpha=0.5)

plt.show()