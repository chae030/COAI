import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

f = open('BTC_USD_2019-02-28_2020-02-27-CoinDesk.csv', 'r')
coindesk_data = pd.read_csv(f, header = 0)
seq = coindesk_data[['Closing Price (USD)']].to_numpy()
# print('데이터 길이:', len(seq), '\n앞쪽 5개 값:', seq[0:5])

# plt.plot(seq, color='red')
# plt.title('Bitcoin Prices (1 year from 2019-02-28)')
# plt.xlabel('Days');plt.ylabel('Price in USD')
# plt.show()

def seq2dataset(seq, window, horizon) :
    X = []; Y = []
    for i in range(len(seq) - (window+horizon) + 1) :
        x = seq[i:(i+window)]
        y = (seq[i+window+horizon-1])
        X.append(x); Y.append(y)
    return np.array(X), np.array(Y)

w = 7
h = 1

X, Y = seq2dataset(seq, w, h)
# print(X.shape, Y.shape)
# print(X[0], Y[0]); print(X[-1], Y[-1])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

split = int(len(X) * 0.7)
# 374page