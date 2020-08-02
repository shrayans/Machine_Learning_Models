#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 23:09:43 2020

@author: shrayans
"""
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


class KNNClassifier:
    
    
    k=0
    path=""

    def fun(self):  
         pass
    
    def train(self,p):
        self.path=p
        print(p)
    

#    ds1=pd.read_csv("test_labels.csv",header=None)
#    ds2=pd.read_csv("test.csv",header=None)
#    ds3=pd.read_csv("train.csv",header=None)
#    ds3 = ds3.sample(frac=1).reset_index(drop=True)
#    
#    train_label_df=ds3.iloc[0:,0]
#    train_data_df=ds3.iloc[0:,1:]    
    
    
    def predict(self,test_path):
        ds2=pd.read_csv(test_path,header=None)
        ds3=pd.read_csv(self.path,header=None)
        ds3 = ds3.sample(frac=1).reset_index(drop=True)
        
        train_label_df=ds3.iloc[0:,0]
        train_data_df=ds3.iloc[0:,1:]  
        
        train_data_df.drop(train_data_df.iloc[:,10:11],inplace=True,axis=1)
        ds2.drop(ds2.iloc[:,10:11],inplace=True,axis=1)
        #pd.get_dummies(train_data_df,)
        
        attri=[['s','k','f','x','c','b'],['f','g','y','s'],['n','b','c','g','r','p','u','e','w','y'],['f','t'],['a','l','c','y','f','m','n','p','s'],['n','f','d','a'],['c','w','d'],['n','b'],['k','n','b','h','g','r','o','p','u','e','w','y'],['t','e'],['s','k','y','f'],['f','y','k','s'],['y','w','e','p','o','g','c','b','n'],['y','w','e','p','o','g','c','b','n'],['u','p'],['n','o','w','y'],['t','o','n'],['l','f','e','c','n','p','s','z'],['y','w','u','o','r','h','b','n','k'],['y','v','s','n','c','a'],['g','l','m','p','u','w','d']]
        
        train_data = pd.DataFrame(columns=None)
        test=pd.DataFrame(columns=None)
        
        sum=0
        for i in range(0,21):

            sum+=len(attri[i])
            dummies=pd.get_dummies(train_data_df.iloc[:,i])
            dummies = dummies.T.reindex(attri[i]).T.fillna(0)
            train_data=pd.concat([train_data, dummies], axis=1)
        
        print(sum)
        
        for i in range(0,21):
            sum+=len(attri[i])
            dummies=pd.get_dummies(ds2.iloc[:,i])
            dummies = dummies.T.reindex(attri[i]).T.fillna(0)
            test=pd.concat([test, dummies], axis=1)
        

        test=pd.DataFrame(test).to_numpy()
        
        train_data=pd.DataFrame(train_data).to_numpy()
        train_label=pd.DataFrame(train_label_df).to_numpy()
            
        k=3
        rslt=[]        
        
        for j in range(0,(ds2.shape[0])):
            
            list=[]
            rank = [0 for  i in range(0, 2)]
            
            for i in range(0,train_data.shape[0]):
                d=distance.euclidean(test[j], train_data[i][:])
                temp=[d,train_label[i]]
                list.append(temp)
            
            list.sort()
            
            for a in range(0,k):
                if(list[a][1]=='e'):
                    rank[0]+=1
                else:
                    rank[1]+=1
            
            if(rank[0]>rank[1]):
                maxpos='e'
            else:
                maxpos='p'
               
            rslt.append(maxpos)
        return rslt

#            if(maxpos==test_labels[j]):
#                c+=1
#        #        print("correct prediction as ",maxpos,test_labels[j])
#            else:
#                w+=1
#        #        print("Wrong prediction ",maxpos,test_labels[j])
        
#        print("correct ",c)
#        print("wrong ",w)

knn_classifier = KNNClassifier()
knn_classifier.train('./Datasets/q2/train.csv')
predictions = knn_classifier.predict('./Datasets/q2/test.csv')
test_labels = list()
with open("./Datasets/q2/test_labels.csv") as f:
  for line in f:
    test_labels.append(line.strip())
#print(test_labels, predictions)
print (accuracy_score(test_labels, predictions))    

