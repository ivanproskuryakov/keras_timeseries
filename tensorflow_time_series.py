# url https://www.tensorflow.org/tutorials/structured_data/time_series


import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import tensorflow as tf

from window_generator import WindowGenerator

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

csv_path = "data/jena_climate_2009_2016.csv"
df = pd.read_csv(csv_path)

# Slice [start:stop:step], starting from index 5 take every 6th record.
df = df[5::6]

date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

# plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
# plot_features = df[plot_cols]
# plot_features.index = date_time
# _ = plot_features.plot(subplots=True)
#
# plot_features = df[plot_cols][:480]
# plot_features.index = date_time[:480]
# _ = plot_features.plot(subplots=True)

# print(df.describe().transpose())


wv = df['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = df['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0

# The above inplace edits are reflected in the DataFrame.
df['wv (m/s)'].min()

# plt.hist2d(df['wd (deg)'], df['wv (m/s)'], bins=(50, 50), vmax=400)
# plt.colorbar()
# plt.xlabel('Wind Direction [deg]')
# plt.ylabel('Wind Velocity [m/s]')


wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')

# Convert to radians.
wd_rad = df.pop('wd (deg)') * np.pi / 180

# Calculate the wind x and y components.
df['Wx'] = wv * np.cos(wd_rad)
df['Wy'] = wv * np.sin(wd_rad)

# Calculate the max wind x and y components.
df['max Wx'] = max_wv * np.cos(wd_rad)
df['max Wy'] = max_wv * np.sin(wd_rad)

# plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
# plt.colorbar()
# plt.xlabel('Wind X [m/s]')
# plt.ylabel('Wind Y [m/s]')
# ax = plt.gca()
# ax.axis('tight')

# plt.show()


day = 24 * 60 * 60
year = (365.2425) * day

timestamp_s = date_time.map(pd.Timestamp.timestamp)

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

# plt.plot(np.array(df['Day sin'])[:25])
# plt.plot(np.array(df['Day cos'])[:25])
# plt.xlabel('Time [h]')
# plt.title('Time of day signal')
#
#
# plt.show()


fft = tf.signal.rfft(df['T (degC)'])
f_per_dataset = np.arange(0, len(fft))

n_samples_h = len(df['T (degC)'])
hours_per_year = 24 * 365.2524
years_per_dataset = n_samples_h / (hours_per_year)

f_per_year = f_per_dataset / years_per_dataset

# plt.step(f_per_year, np.abs(fft))
# plt.xscale('log')
# plt.ylim(0, 400000)
# plt.xlim([0.1, max(plt.xlim())])
# plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
# _ = plt.xlabel('Frequency (log scale)')
#
#
# plt.show()


# Split the data

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n * 0.7)]
val_df = df[int(n * 0.7):int(n * 0.9)]
test_df = df[int(n * 0.9):]

num_features = df.shape[1]

# Normalize the data

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# df_std = (df - train_mean) / train_std
# df_std = df_std.melt(var_name='Column', value_name='Normalized')
# plt.figure(figsize=(12, 6))
# ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
# _ = ax.set_xticklabels(df.keys(), rotation=90)

# plt.show()
# Data windowing


# WindowGenerator
# ------------------------------------------------------------------

w1 = WindowGenerator(
    input_width=24,
    label_width=1,
    shift=24,
    label_columns=['T (degC)'],
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
)
print(w1)

w2 = WindowGenerator(
    input_width=6,
    label_width=1,
    shift=1,
    label_columns=['T (degC)'],
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
)
print(w2)



# Stack three slices, the length of the total window.
# ------------------------------------------------------------------

example_window = tf.stack([
    np.array(train_df[:w2.total_window_size]),
    np.array(train_df[100:100 + w2.total_window_size]),
    np.array(train_df[200:200 + w2.total_window_size])
])

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')

# w2.plot(example_inputs, example_labels)
# w2.plot(example_inputs, example_labels, plot_col='p (mbar)')
#
# plt.show()


print(w2.train.element_spec)


for example_inputs, example_labels in w2.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')




# single_step_window
# ------------------------------------------------------------------

single_step_window = WindowGenerator(
    input_width=1,
    label_width=1,
    shift=1,
    label_columns=['T (degC)'],
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
)

for example_inputs, example_labels in single_step_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')