# Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras
# base on tutorial of Jason Brownlee
# source https://github.com/ivanarielcaceres/timeseries-lstm-keras
# https://www.kaggle.com/code/ternaryrealm/lstm-time-series-explorations-with-keras/notebook

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle


SAMPLES = 5000
PERIOD = 50
x = np.linspace(-PERIOD * np.pi, PERIOD * np.pi, SAMPLES)
series = pd.DataFrame(np.sin(x))

plt.figure(figsize = (15, 5))
plt.plot(series.values[:PERIOD])
plt.xlabel("x")
plt.ylabel("y")
plt.title("y = sin(x), first %d samples" % PERIOD)
# plt.show()

# Normalize data on the (-1, 1) interval.
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(series.values)

# Convert to data frame.
series = pd.DataFrame(scaled)


# Helper function to create a windowed data set.
# FIXME: Copying & overwriting is flawed!
def create_window(data, window_size=1):
    data_s = data.copy()
    for i in range(window_size):
        data = pd.concat([data, data_s.shift(-(i + 1))],
                         axis=1)

    data.dropna(axis=0, inplace=True)
    return (data)


# FIXME: We'll use this only for demonstration purposes.
series_backup = series.copy()
t = create_window(series_backup, 1)
t.head()

print(t.head())


window_size = 50
series = create_window(series, window_size)
print("Shape of input data:")
print(series.shape)


# Using 80% of data for training, 20% for validation.
# FIXME: Need to align with example 1.
TRAIN_SIZE = 0.80

nrow = round(TRAIN_SIZE * series.shape[0])

train = series.iloc[:nrow, :]
test = series.iloc[nrow:, :]

# Shuffle training data.
train = shuffle(train)

train_X = train.iloc[:, :-1]
test_X = test.iloc[:, :-1]

train_Y = train.iloc[:, -1]
test_Y = test.iloc[:, -1]

print("Training set shape for X (inputs):")
print(train_X.shape)
print("Training set shape for Y (output):")
print(train_Y.shape)


train_X = np.reshape(train_X.values, (train_X.shape[0], train_X.shape[1], 1))
test_X = np.reshape(test_X.values, (test_X.shape[0], test_X.shape[1], 1))

print(train_X.shape)
print(test_X.shape)


# Define the model.
model2 = Sequential()
model2.add(LSTM(input_shape = (window_size, 1),
               units = window_size,
               return_sequences = True))
model2.add(Dropout(0.5))
model2.add(LSTM(256))
model2.add(Dropout(0.5))
model2.add(Dense(1))
model2.add(Activation("linear"))
model2.compile(loss = "mse",
              optimizer = "adam")
model2.summary()

# Fit the model.
model2.fit(train_X,
          train_Y,
          batch_size = 512,
          epochs = 3,
          validation_split = 0.1)

# Predict on test data.
pred_test = model2.predict(test_X)

# Apply inverse transformation to get back true values.
test_y_actual = scaler.inverse_transform(test_Y.values.reshape(test_Y.shape[0], 1))

print("MSE for predicted test set: %2f" % mean_squared_error(test_y_actual, pred_test))

plt.figure(figsize = (15, 5))
plt.plot(test_y_actual, label="True value")
plt.plot(pred_test, label="Predicted value")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.title("Comparison true vs. predicted test set")
plt.legend()
plt.show()