import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os
import string
from sklearn import svm
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import nltk
nltk.download('punkt')


class AuthorClassifier:
    
  train_path=""
  test_path=""
  
  
  
  def fun(self):
    pass

  def predict(self,s):
      self.test_path=str(s)
      return self.my_fun()
  
  def train(self,s):
      self.train_path=str(s)    

  def my_fun(self):
      

    ds1=pd.read_csv(self.train_path)
    train_labels=ds1.iloc[:,2]
    
    train_row=ds1.shape[0]
    
    ds2=pd.read_csv(self.test_path)
    test_row=ds1.shape[0]

    
    corpus1=ds1.iloc[:,1].to_numpy()
    corpus2=ds2.iloc[:,1].to_numpy()    
    
    corpus=np.concatenate([corpus1,corpus2],axis=0)
    
#    print(corpus.shape)
    
    for i in range(len(corpus)):
    
      regex = re.compile('[^a-zA-Z]')
      corpus[i]=regex.sub(' ', corpus[i])
      txt = ''.join(corpus[i])
      corpus[i]=txt

        
    
    for i in range(len(corpus)):
      
      txt1=corpus[i].split(' ')
      txt=""
      
      for j in txt1:
          if(len(j)>3):
              txt+=" "+j
      corpus[i]=txt
    
    
    #    clf = svm.SVC(kernel='linear',C=1)
#    clf.fit(train_data, train_labels)
#    prediction=clf.predict(test_data)
#    return prediction

    
    
    
    for i in range(len(corpus)):
    
      stemmer= PorterStemmer()
    
      txt1=word_tokenize(corpus[i])
      txt=""
      for word in txt1:
          txt+=" "+stemmer.stem(word)
    
      corpus[i]=txt
    corpus
    
    
    
    my_stop_words = text.ENGLISH_STOP_WORDS

    
    
    vectorizer = TfidfVectorizer(stop_words=my_stop_words)
    X = vectorizer.fit_transform(corpus)
    X=X.toarray()

#    print(X,X.shape)
    
    
    
    #from sklearn.decomposition import PCA
    #pca = PCA(n_components=1000)
    #X=pca.fit_transform(X)
    
    
    train_data=X[:train_row]
    test_data=X[train_row:]
    
#    
#    
#    print("Train data shape:- ",train_data.shape)
#    print("Train labels shape:- ",test_data.shape)
#    

    clf = svm.SVC(kernel='linear',C=1)
    clf.fit(train_data, train_labels)
    prediction=clf.predict(test_data)
    return prediction







#from q5 import AuthorClassifier as ac
auth_classifier = AuthorClassifier()
auth_classifier.train('Datasets/Question-5/Train(1).csv') # Path to the train.csv will be provided
predictions = auth_classifier.predict('Datasets/Question-5/Train(1).csv') # Path to the test.csv will be provided

