import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_PATH = './data/data.csv'

def window_data(data, window=3):
    last_win_data = data.iloc[-window:,:].values
    last_win_data = last_win_data.reshape((1,)+last_win_data.shape)
    result = []
    for index in range(len(data)-window):
        result.append(np.array(data[index:index+window+1]))
    result = np.array(result)
    x_data = result[:,:-1,:]
    y_data = result[:,-1,-1]
    return x_data, y_data, last_win_data

def load_training_data(LAG=1,window=3,val=18):
    data = pd.read_csv(DATA_PATH)
    if LAG>0:
        data.iloc[:,:-1] = data.iloc[:,:-1].shift(LAG)
        data.dropna(axis=0, inplace=True)
    if val!=0:
        train = data.iloc[:-val,:].copy()
    else:
        train = data.iloc[:,:].copy()
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train)
    x_train, y_train, last_win_data = window_data(data=pd.DataFrame(train_norm), window=window)
    return x_train, y_train, scaler, data, last_win_data

def load_validation_data(data,scaler,window=3,val=18):
    test = data.iloc[-val:,:].copy()
    test_norm = scaler.transform(test)
    x_test, y_test, last_win_data = window_data(data=pd.DataFrame(test_norm), window=window)
    return x_test, y_test