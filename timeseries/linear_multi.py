import IPython
import tensorflow as tf
import matplotlib.pyplot as plt

from base import multi_window, compile_and_fit, num_features

# Linear Multi
# ------------------------------------------------------------------
# A simple linear model based on the last input time step does better than
# either baseline, but is underpowered. The model needs to predict OUTPUT_STEPS
# time steps, from a single input time step with a linear projection. It can only
# capture a low-dimensional slice of the behavior, likely based mainly
# on the time of day and time of year.

OUT_STEPS = 24

multi_linear_model = tf.keras.Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_linear_model, multi_window)

multi_val_performance = {}
multi_performance = {}


IPython.display.clear_output()
multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_linear_model)


plt.show()