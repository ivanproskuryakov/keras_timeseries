# url https://www.tensorflow.org/tutorials/structured_data/time_series

import IPython
import tensorflow as tf
import matplotlib.pyplot as plt

from base import compile_and_fit, conv_window

# 3. Multi-step dense
# ------------------------------------------------------------------
# A single-time-step model has no context for the current values of its inputs.
# It can't see how the input features are changing over time. To address this
# issue the model needs access to multiple time steps when making predictions


conv_window.plot()

# plt.title("Given 3 hours of inputs, predict 1 hour into the future.")
# plt.show()


multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])


print('Input shape:', conv_window.example[0].shape)
print('Output shape:', multi_step_dense(conv_window.example[0]).shape)


history = compile_and_fit(multi_step_dense, conv_window)


val_performance = {}
performance = {}

IPython.display.clear_output()
val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)


conv_window.plot(multi_step_dense)

plt.show()