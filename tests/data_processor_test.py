import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import data_processor 
from models import ModelRunner, Baseline
import yfinance as yf
import numpy as np
import tensorflow as tf


aapl = yf.Ticker('AAPL')

df = aapl.history(start='2000-01-01', end='2020-01-01')

pp = data_processor.PreProcessor(df, date='Date', open='Open', high='High', low='Low', close='Close', volume='Volume')

df = pp.process()

print (pp.df.columns)

print("==================")

print(df.columns)

print("***************************************")

train, validate, test = np.split(df, [int(.6*len(df)), int(.8*len(df))])
input_width = 5
label_width = 1
shift = 1
y = ['Close_pct_change_log']
wp = data_processor.WindowGenerator(input_width, label_width, shift, train_df=train, val_df=validate, test_df=test, label_columns=y)

for example_inputs, example_labels in wp.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')

bl = Baseline()

dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(3,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

mr = ModelRunner()
val_performance = {}
performance = {}

model = conv_model
history = mr.compile_and_fit(model, wp)

val_performance['Model'] = model.evaluate(wp.val)
performance['Model'] = model.evaluate(wp.test, verbose=0)

for name, value in performance.items():
  print(f'{name:15s}: {value[1]:0.4f}')
