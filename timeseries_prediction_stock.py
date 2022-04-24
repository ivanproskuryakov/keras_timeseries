# https://www.kaggle.com/code/amarpreetsingh/stock-prediction-lstm-using-keras

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


data = pd.read_csv('./data/all_stocks_5yr.csv')
cl = data[data['Name']=='MMM'].Close


scl = MinMaxScaler()
cl = cl.values.reshape(cl.shape[0], 1)
cl = scl.fit_transform(cl)

def processData(data, lb):
    X,Y = [],[]

    for i in range(len(data)-lb-1):
        X.append(data[i:(i+lb),0])
        Y.append(data[(i+lb),0])

    return np.array(X), np.array(Y)

X,y = processData(cl, 7)

print('X')
print(X)
print(len(X))
print('y')
print(y)
print(len(y))
exit()

X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]

#Reshape data for (Sample, Timestep, Features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],1))

#Build the model
model = Sequential()

# model.add(LSTM(256,input_shape=(7,1)))
model.add(LSTM(6,input_shape=(7,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

#Fit model with history to check for overfitting
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    shuffle=False
)

Xt = model.predict(X_test)

# plt.plot(scl.inverse_transform(y_test.reshape(-1,1)))
# plt.plot(scl.inverse_transform(Xt))


plt.show()
