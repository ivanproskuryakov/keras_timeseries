# url https://www.tensorflow.org/tutorials/structured_data/time_series

import IPython
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from base import compile_and_fit, wide_window, multi_window

# 6. Multi-step models
# ------------------------------------------------------------------
# Both the single-output and multiple-output models in the previous sections made
# single time step predictions, one hour into the future.


multi_window.plot()

plt.show();
