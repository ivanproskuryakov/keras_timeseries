# url https://www.tensorflow.org/tutorials/structured_data/time_series

import IPython
import tensorflow as tf
import matplotlib.pyplot as plt

from base import compile_and_fit, num_features, multi_window

# RNN
# ------------------------------------------------------------------

# A recurrent model can learn to use a long history of inputs, if it's relevant
# to the predictions the model is making. Here the model will accumulate internal
# state for 24 hours, before making a single prediction for the next 24 hours.

# In this single-shot format, the LSTM only needs to produce an output at the last
# time step, so set return_sequences=False in tf.keras.layers.LSTM.

OUT_STEPS = 24
CONV_WIDTH = 3

val_performance = {}
performance = {}
multi_val_performance = {}
multi_performance = {}

multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features].
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_lstm_model, multi_window)

IPython.display.clear_output()

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model)

plt.show()
