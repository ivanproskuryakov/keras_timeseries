# url https://www.tensorflow.org/tutorials/structured_data/time_series

import tensorflow as tf
import matplotlib.pyplot as plt

from base import single_step_window, wide_window
from linear import compile_and_fit

# 2. Dense
# ------------------------------------------------------------------
# Before applying models that actually operate on multiple time-steps,
# it's worth checking the performance of deeper, more powerful, single input step models.
# Here's a model similar to the linear model, except it stacks several a few Dense
# layers between the input and the output:

dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

history = compile_and_fit(dense, single_step_window)

val_performance = {}
performance = {}

val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)


wide_window.plot(dense)

plt.show()
