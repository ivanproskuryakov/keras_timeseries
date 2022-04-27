# url https://www.tensorflow.org/tutorials/structured_data/time_series

import IPython
import tensorflow as tf
import matplotlib.pyplot as plt

from base import compile_and_fit, num_features, multi_window

# CNN
# ------------------------------------------------------------------

# A convolutional model makes predictions based on a fixed-width history,
# which may lead to better performance than the dense model since it can see
# how things are changing over time:


OUT_STEPS = 24
CONV_WIDTH = 3

val_performance = {}
performance = {}
multi_val_performance = {}
multi_performance = {}

multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_conv_model, multi_window)

IPython.display.clear_output()

multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_conv_model)

plt.show()
