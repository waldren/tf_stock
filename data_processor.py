import os
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import tensorflow as tf
'''
Source: https://www.tensorflow.org/tutorials/structured_data/time_series
'''


class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=None, val_df=None, test_df=None,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]]
            for name in self.label_columns],
            axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

  # WindowGenerator.split_window = split_window
  '''
  def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(
                plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()
  '''
  # WindowGenerator.plot = plot

  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)

    ds = ds.map(self.split_window)

    return ds

  # WindowGenerator.make_dataset = make_dataset

  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result

  # WindowGenerator.train = train
  # WindowGenerator.val = val
  # WindowGenerator.test = test
  # WindowGenerator.example = example

class PreProcessor:
    def __init__(self, dataframe, date='dt', open='open', high='high', low='low', close='close', volume='volume'):
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.df = dataframe
        self.features = []
    
    def process(self):
        # Run the entire preprocessing process
        #create a returns columns
        self.create_pct_change_log(columns=[self.close])

        #setup some moving averages
        self.create_movingaverage(columns=[self.close, self.volume])
        self.create_movingaverage(columns=[self.close], period=20, postfix='_20ma')
        self.create_pct_change_log(columns=["{}_10ma".format(self.close),"{}_10ma".format(self.volume), "{}_20ma".format(self.close)])
        
        #create true ranges for 10 and 20 bars
        self.create_true_ranges(periods=[10,20])

        # create features for the high and low range percent difference
        self.pct_diff_log(column1=self.high, column2=self.low)

        # create features for the difference between close and the moving averages
        self.pct_diff_log(column1="{}_10ma_pct_change_log".format(self.close), column2="{}_pct_change_log".format(self.close))
        self.pct_diff_log(column1="{}_20ma_pct_change_log".format(self.close), column2="{}_pct_change_log".format(self.close))

        return self.get_feature_dataframe()
    
    def create_pct_change_log(self, columns=['open','close'], postfix='_pct_change_log'):
        for c in columns:
            f = c + postfix
            self.df[f] = np.log (1+ self.df[c].pct_change())
            self.features.append(f)
    
    def create_movingaverage(self, columns=['open','close'], period=10, postfix='_10ma'):
        for c in columns:
            f = c + postfix
            self.df[f] = self.df[c].rolling(window=period).mean()
    
    def create_true_ranges(self, periods=[10]):
        self.df['TR'] = abs(self.df[self.high] - self.df[self.low])
        self.create_pct_change_log(columns=['TR'])
        for period in periods:
            f = "ATR_{}p".format(period)
            self.df[f] = self.df['TR'].rolling(window=period).mean()
            self.create_pct_change_log(columns=[f])


    def pct_diff_log(self, column1='high', column2='low'):
        f = "{}_{}_pct_diff_log".format(column1,column2)
        self.df[f] = np.log(1+(self.df[column1] / self.df[column2] - 1))
        self.features.append(f)

    def get_feature_dataframe(self):
        columns = []
        # columns.append(self.open)
        # columns.append(self.high)
        # columns.append(self.low)
        # columns.append(self.close)
        # columns.append(self.volume)
        for f in self.features:
            columns.append(f)
        # Drop any rows that do not have all the features (should drop rows = largest moving average period)
        return self.df[columns].dropna(axis=0)
        
