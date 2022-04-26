# url https://www.tensorflow.org/tutorials/structured_data/time_series

import IPython
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from base import compile_and_fit, wide_window

# 6. Recurrent neural network
# ------------------------------------------------------------------
# A Recurrent Neural Network (RNN) is a type of neural network well-suited to time series data.
# RNNs process a time series step-by-step, maintaining an internal state from time-step to time-step.

val_performance = {}
performance = {}

# With return_sequences=True, the model can be trained on 24 hours of data at a time.
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])


print('Input shape:', wide_window.example[0].shape)
print('Output shape:', lstm_model(wide_window.example[0]).shape)


history = compile_and_fit(lstm_model, wide_window)

IPython.display.clear_output()
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)


wide_window.plot(lstm_model)

x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [T (degC), normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)

_ = plt.legend()

plt.show()
