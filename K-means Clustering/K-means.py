from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os
import string
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import distance
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from nltk.stem import WordNetLemmatizer

class Cluster:
    
  train_path=""
  test_path=""

  def fun(self):
    pass

  def cluster(self,s):
#    print("yoo")  
    path=s
    
    list=os.listdir(path)
#    list.sort()
    
    k=5
  
    labels=[]
    
        
    corpus=[]
#    print("\n\n")
    
    
    for i in range (len(list)):
        
        file = open(path+"/"+list[i],mode='rb')
         
        all_of_it = file.read().decode('utf-8',errors='ignore')
        
        file.close()
        
        txt=all_of_it
        
        regex = re.compile('[^a-zA-Z]')
        txt=regex.sub(' ', txt)
        txt = ''.join(txt)
        
        
        txt1=txt.split(' ')
        txt="" 
        for i in txt1:
            if(len(i)>3 ):
                txt+=" "+i
    
        
        
        stemmer= PorterStemmer()
    
        txt1=word_tokenize(txt)
        txt=""
        for word in txt1:
            txt+=" "+stemmer.stem(word)
        
        
        lemmatizer=WordNetLemmatizer()
        
        txt1=word_tokenize(txt)
        txt=""
        for word in txt1:
            txt+=" "+lemmatizer.lemmatize(word)
        corpus.append(txt)
        
    
    my_stop_words = text.ENGLISH_STOP_WORDS
    
    
    vectorizer = TfidfVectorizer(stop_words=my_stop_words)
    X = vectorizer.fit_transform(corpus)
    
#    feature=vectorizer.get_feature_names()
    
    X=X.toarray()
    
        
    n=X.shape[0]
    c=X.shape[1]
    

#    index=np.random.choice(1725,5,replace=False)
    centers=X[np.random.choice(n,5,replace=False)]
    print(centers.shape)

    print(centers.shape)

    dict_centroid={}
    
    
    for itr in range(5):
    
        for j in range(k):
            temp=[]
            dict_centroid[j]=temp    
        
        
        for i in range(n):
            
            distance_array=[]
            
            for j in range(k):
                distance_array.append(distance.euclidean(X[i], centers[j] ))
            
            minpos = distance_array.index(min(distance_array))
                     
            dict_centroid[minpos].append(X[i])
            
        print(itr,end="-> ")    
        
        for j in range(k):
            print(len(dict_centroid[j]),end=' ')
            centers[j]=np.mean(np.asarray(dict_centroid[j]),axis=0)
            
        print("")
        
    labels=[]
    
    for i in range(n):
        
        distance_array=[]
        
        for j in range(k):
            distance_array.append(distance.euclidean(X[i], centers[j] ))
        
        minpos = distance_array.index(min(distance_array))

        labels.append(minpos+1)
        
#    print(centers)

    final_dict={}
    
    for i in range(n):
        final_dict[list[i]]=labels[i]

    return final_dict





#for i in range(n-1600):
#    plt.scatter(X[i, :], X[i,:], s=7, color ='green')
#    print(X[i,0],X[i,1])


#print(feature,len(feature))
#print('\n\n\n')
#
#print(X,len(X[0]))

#for i in range(len(X[0])):
#    print(feature[i],X[0][i])
from q6 import Cluster as cl
cluster_algo = cl()
# You will be given path to a directory which has a list of documents. You need to return a list of cluster labels for those documents
predictions = cluster_algo.cluster('Datasets/Question-6/dataset') 
print(predictions)