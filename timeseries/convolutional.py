# url https://www.tensorflow.org/tutorials/structured_data/time_series

import IPython
import tensorflow as tf
import matplotlib.pyplot as plt

from base import conv_window, compile_and_fit, wide_window, wide_conv_window

# 5. Convolution neural network
# ------------------------------------------------------------------
# A convolution layer (tf.keras.layers.Conv1D) also takes multiple time steps as input to each prediction.
# Below is the same model as multi_step_dense, re-written with a convolution.

# Note the changes:
# The tf.keras.layers.Flatten and the first tf.keras.layers.Dense are replaced by a tf.keras.layers.Conv1D.
# The tf.keras.layers.Reshape is no longer necessary since the convolution keeps the time axis in its output.

val_performance = {}
performance = {}

conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=(3,),
        activation='relu'
    ),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

print("Conv model on `conv_window`")
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)

history = compile_and_fit(conv_model, conv_window)
IPython.display.clear_output()

val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)


print("Wide window")
print('Input shape:', wide_window.example[0].shape)
print('Labels shape:', wide_window.example[1].shape)
print('Output shape:', conv_model(wide_window.example[0]).shape)


print("Wide conv window")
print('Input shape:', wide_conv_window.example[0].shape)
print('Labels shape:', wide_conv_window.example[1].shape)
print('Output shape:', conv_model(wide_conv_window.example[0]).shape)


wide_conv_window.plot(conv_model)

plt.show()