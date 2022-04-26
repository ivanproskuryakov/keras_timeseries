import tensorflow as tf
import matplotlib.pyplot as plt

from base import single_step_window, wide_window, train_df, compile_and_fit

# 2. Linear model
# ------------------------------------------------------------------
# The simplest trainable model you can apply to this task is to insert linear
# transformation between the input and output. In this case the output from a time
# step only depends on that step:


linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])

print('Input shape:', single_step_window.example[0].shape)
print('Output shape:', linear(single_step_window.example[0]).shape)

history = compile_and_fit(linear, single_step_window)


val_performance = {}
performance = {}

val_performance['Linear'] = linear.evaluate(single_step_window.val)
performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)


plt.bar(x = range(len(train_df.columns)),
        height=linear.layers[0].kernel[:,0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
_ = axis.set_xticklabels(train_df.columns, rotation=90)

wide_window.plot(linear)

plt.show()
