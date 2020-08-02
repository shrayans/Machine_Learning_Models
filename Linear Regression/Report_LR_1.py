
class Airfoil:

  train_path=""
  test_path=""


  def fun(self):
    pass

  def train(self,s):
      self.train_path=str(s)

  def my_fun(self):

    import pandas as pd
    ds1=pd.read_csv(str(self.train_path),header=None)
    
    
    import numpy as np
    train_num=ds1.iloc[:,:5].to_numpy() 
    train_labeling=ds1.iloc[:,5:6].to_numpy()
    
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train_num)
    train_num=scaler.transform(train_num)

    
    ds=pd.DataFrame(np.ones((ds1.shape[0],), dtype=int))
    n=ds.to_numpy()
    train_num=np.append(n, train_num, axis=1)
    
    train_data=train_num[:,:]
    train_labels=train_labeling[:]
    
    
#    Extracting test data
    
    ds1=pd.read_csv(str(self.test_path),header=None)
    ds1=ds1.iloc[:,:].to_numpy()
    scaler = MinMaxScaler()
    scaler.fit(ds1)
    ds1=scaler.transform(ds1)
    
    ds=pd.DataFrame(np.ones((ds1.shape[0],), dtype=int))
    n=ds.to_numpy()
    test_data=np.append(n,ds1,axis=1)
   
#    test_data=train_num[:,:]
#    test_labels=train_labeling[:]
#    test_labels.shape
#    
    
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    
    theta=np.random.randn(6,1)
    
    cost_array=[]
    itr_array=[]
    
    m=len(train_labels[:])
    
    
    for i in range(100000):
        
      h=np.dot(train_data,theta)
    
      theta=theta-(1/m)*0.1*(train_data.T.dot((h-train_labels)))
    
      cost=mean_squared_error(h,train_labels)
      cost_array.append(cost)
      itr_array.append(i)
    
    

    h=np.dot(test_data,theta)

    
#    print("MSE:- ",mean_squared_error(h,train_labels))
#    print("r^2:- ",r2_score(train_labels,h))
    
    return h

  def predict(self,s):
      self.test_path=str(s)
      return self.my_fun()

#from q3 import Airfoil as ar
model3 = Airfoil()
model3.train("/home/shrayans/SMAI/assi2/Datasets/Question-3/airfoil/airfoil.csv") # Path to the train.csv will be provided
prediction3 = model3.predict('/home/shrayans/SMAI/assi2/Datasets/Question-3/airfoil/airfoil.csv') # Path to the test.csv will be provided