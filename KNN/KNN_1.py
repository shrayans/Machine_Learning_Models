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
      

    def checkIfDuplicates_1(self,listOfElems):
        if (len(listOfElems) == len(set(listOfElems))):
            return False
        else:
            return True
        
        
    
    def predict(self,test_path):
#        ds1=pd.read_csv("test_labels.csv",header=None)
        ds2=pd.read_csv(test_path,header=None)
#        ds3=pd.read_csv("train.csv",header=None)
        ds3=pd.read_csv(self.path,header=None)
        
#        test_labels=pd.DataFrame(ds1).to_numpy()
        test=pd.DataFrame(ds2).to_numpy()
        train=pd.DataFrame(ds3).to_numpy()

        rslt=[]
        
        k=5
                
        for j in range(0,(ds2.shape[0])):
            
            list=[]
            rank = [-i for  i in range(0, 10)]
            
#            print(rank)
            
            for i in range(0,ds3.shape[0]):
                d=distance.euclidean(test[j], train[i][1:])
                temp=[d,train[i][0]]
                list.append(temp)
            
            list.sort()            
            for a in range(0,k):
                if(rank[list[a][1]]<0):
                    rank[list[a][1]]=1
                else:
                    rank[list[a][1]]+=1
                    
#            print(rank)
            
            maxpos = rank.index(max(rank))
            
            if(self.checkIfDuplicates_1(rank)==True):
                rslt.append(list[0][1])
            else:
                rslt.append(maxpos)
                
        return rslt

#            if(maxpos==test_labels[j]):
#                c+=1
#                print("correct prediction as ",list[0][1],"->",test_labels[j])
#            else:
#                w+=1
#                print("Wrong prediction ",list[0][1],"->",test_labels[j])
        
#        print("correct ",c)
#        print("wrong ",w)
#        
knn_classifier = KNNClassifier()
#
knn_classifier.train('./Datasets/q1/train.csv')

predictions = knn_classifier.predict('./Datasets/q1/test.csv')
test_labels = list()
with open("./Datasets/q1/test_labels.csv") as f:
  for line in f:
    test_labels.append(int(line))
#print(test_labels,predictions)
print (accuracy_score(test_labels, predictions))    