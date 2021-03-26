import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import data_processor 
import yfinance as yf
import numpy as np

aapl = yf.Ticker('AAPL')

df = aapl.history(start='2017-01-01', end='2020-01-01')

pp = data_processor.PreProcessor(df, date='Date', open='Open', high='High', low='Low', close='Close', volume='Volume')

df = pp.process()

print (pp.df.columns)

print("==================")

print(df.columns)

print("***************************************")

train, validate, test = np.split(df, [int(.6*len(df)), int(.8*len(df))])
input_width = 30
label_width = 1
shift = 1
y = ['Close_pct_change_log']
wp = data_processor.WindowGenerator(input_width, label_width, shift, train_df=train, val_df=validate, test_df=test, label_columns=y)

ds = wp.example()


