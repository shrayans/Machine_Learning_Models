import numpy as np
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

class Weather:
    
  train_path=""
  test_path=""

  def fun(self):
    pass

  def train(self,s):
      self.train_path=str(s)

  def predict(self,s):
      self.test_path=str(s)
      return self.my_fun()
    
  def my_fun(self):

    import pandas as pd
    ds1=pd.read_csv(self.train_path)
    ds2=pd.read_csv(self.test_path)
    
    
#    ds2=ds2.drop(['Apparent Temperature (C)'], axis=1)    
    
    ds1=ds1.iloc[:,1:]
    ds2=ds2.iloc[:,1:]
    
    
    ds1['Precip Type'].fillna("No", inplace = True)
    ds2['Precip Type'].fillna("No", inplace = True)
    

       
    t_np1=pd.get_dummies(ds1["Precip Type"])
    t_np1=t_np1.to_numpy()
    t_np1
        
    
    
    train_labels=ds1.iloc[0:,3].to_numpy()
    train_labels=np.array([train_labels])
    train_labels=train_labels.T
        
    ds1=ds1.drop(['Apparent Temperature (C)','Daily Summary','Summary','Precip Type'], axis=1)
    t_np3=ds1.to_numpy()
    
    
    data=np.concatenate((t_np1,t_np3), axis=1)
#    print(data,data.shape)
    
    train_data=data[0:,:]

       
    t_np1=pd.get_dummies(ds2["Precip Type"])
    t_np1=t_np1.to_numpy()
    t_np1
    
    ds2=ds2.drop(['Daily Summary','Summary','Precip Type'], axis=1)
    t_np3=ds2.to_numpy()
    
    
    data=np.concatenate((t_np1,t_np3), axis=1)
#    print(data,data.shape)
    
    test_data=data[:,:]
#    print(train_data.shape, test_data.shape)
    
    
    
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_data=scaler.transform(train_data)
    
    scaler = MinMaxScaler()
    scaler.fit(test_data)
    test_data=scaler.transform(test_data)
    
    
    from sklearn.metrics import mean_absolute_error
    theta=np.random.randn(train_data.shape[1],1)
    
    alpha=.1
     
    
    for i in range(1000):
    
      m=len(train_labels)
      h=np.dot(train_data,theta)
    
      theta=theta-(1/m)*alpha*(train_data.T.dot((h-train_labels)))
    
    
    h=np.dot(test_data,theta)
    return h
#    print("mse-> ",mean_squared_error(h,train_labels))
#    print("mae-> ",mean_absolute_error(train_labels,h))
#    print("r2-> ",r2_score(train_labels,h))


#from q4 import Weather as wr
model4 = Weather()
model4.train("/home/shrayans/SMAI/assi2/Datasets/Question-4/weather.csv") # Path to the train.csv will be provided 
prediction4 = model4.predict("/home/shrayans/SMAI/assi2/Datasets/Question-4/weather.csv") # Path to the test.csv will be provided
# prediction4 should be Python 1-D List
'''WE WILL CHECK THE MEAN SQUARED ERROR OF PREDICTIONS WITH THE GROUND TRUTH VALUES'''