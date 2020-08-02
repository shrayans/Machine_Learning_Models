import pandas as pd
import numpy as np

import sys

argumentList = sys.argv

# df=pd.read_csv("/content/drive/My Drive/ass3/Q4/household_power_consumption.txt",sep=";")
df=pd.read_csv(argumentList[1],sep=";")
df=df["Global_active_power"]


missing_index=[]
for i in range(df.shape[0]):
  if(df[i]=='?'):
    missing_index.append(i)


df=df.replace('?',np.NaN)

df_train=df[pd.notnull(df)]



n_array=df_train.to_numpy()
n_array = n_array.astype(np.float)
print(n_array.shape)


def time_series(data, time):
  data_item = []
  label = []
  for i in range(len(data)-time):
    x= data[i:i+time]
    y= data[i+time]
    data_item.append(x)
    label.append(y)
  return np.array(data_item), np.array(label)



from sklearn.model_selection import train_test_split

def split_data(train_data, train_label):
   X_train, X_valid, y_train, y_valid = train_test_split(
       train_data, train_label, test_size=0.33, random_state=42)
   return X_train, X_valid, y_train, y_valid



train_data,train_labels=time_series(n_array,60)
X_train, X_valid, y_train, y_valid = split_data(train_data,train_labels)
print(X_train.shape,y_train.shape,X_valid.shape)


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(128, activation='relu', input_dim=60 ) )
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='tanh'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse',metrics=['mse'])

model.fit(X_train, y_train, epochs=15)

prediction = model.predict(X_valid)

cal_perfromance(prediction , y_valid)



original_array=df.to_numpy()
original_array = original_array.astype(np.float)

l=[]

for i in range(len(missing_index)):
  # print(original_array[missing_index[i]-60:missing_index[i]+1])
  temp=original_array[missing_index[i]-60:missing_index[i]]
  temp=temp.reshape(1,60)
  prediction = model.predict(temp)
  original_array[missing_index[i]]=prediction[0][0]
  print(prediction[0][0])
  # l.append(prediction[0][0])

