# url https://www.tensorflow.org/tutorials/structured_data/time_series

import tensorflow as tf
import matplotlib.pyplot as plt

from base import multi_window

class RepeatBaseline(tf.keras.Model):
  def call(self, inputs):
    return inputs


multi_val_performance = {}
multi_performance = {}

repeat_baseline = RepeatBaseline()
repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test, verbose=0)
multi_window.plot(repeat_baseline)


plt.show()