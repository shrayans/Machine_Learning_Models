import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import numpy as np
import os
from PIL import Image,ImageOps
from sklearn.preprocessing import StandardScaler
import random 
import sys

def sigmoid(prediction):
    g=1+np.exp(-prediction)
    return np.power(g,-1)

argumentList = sys.argv 

fo = open(argumentList[1], "r+")
line = fo.readlines()


list=[]
names=[]

for l in line:
	l=l.strip()
	l=l.split(' ')
	list.append(l[0])
	names.append(l[1])

# print(list,names)

fo = open(argumentList[2], "r+")
line = fo.readlines()


test_list=[]

for l in line:
	l=l.strip()
	test_list.append(l)

# print(test_list)

train_images=[]

for i in range(0,len(list)):
    img = Image.open(list[i])
    img.thumbnail((64, 64), Image.ANTIALIAS)
    img = ImageOps.grayscale(img)
    img=np.array(img)
    train_images.append(img)
    
train_images=np.array(train_images)


test_images=[]

for i in range(len(test_list)):
    img = Image.open(test_list[i])
    img.thumbnail((64, 64), Image.ANTIALIAS)
    img = ImageOps.grayscale(img)
    img=np.array(img)
    test_images.append(img)
    
test_images=np.array(test_images)

# print(train_images.shape, test_images.shape)




flatten_images_train=[]
for i in range(len(train_images)):
    flatten_images_train.append(train_images[i].flatten())
flatten_images_train=np.array(flatten_images_train)
flatten_images_train=flatten_images_train/255


flatten_images_test=[]
for i in range(len(test_images)):
    flatten_images_test.append(test_images[i].flatten())
flatten_images_test=np.array(flatten_images_test)
flatten_images_test=flatten_images_test/255



from sklearn.decomposition import PCA
pca=PCA(9)
pca_images_train=pca.fit(flatten_images_train)

pca_images_train=pca.transform(flatten_images_train)

pca_images_test=pca.transform(flatten_images_test)

pca_images_train=np.insert(pca_images_train,0,1,axis=1)
pca_images_test=np.insert(pca_images_test,0,1,axis=1)
# print(pca_images_train.shape,pca_images_test.shape)



labels=[]
dict_names={}
for i in range(len(list)):
    img=list[i].split('/')
    t=img[2].split('_')
    dict_names[int(t[0])]=names[i]

    labels.append(int(t[0]))
labels=np.array(labels)
labels=labels.reshape(len(list),1)
# print(labels,dict_names)


unique_class=np.unique(labels)
from sklearn.preprocessing import OneHotEncoder 
onehotencoder = OneHotEncoder() 
onehotencoder.fit(labels)
encoded_labels = onehotencoder.transform(labels).toarray()

train_encoded_labels=encoded_labels
# test_encoded_labels=encoded_labels[420:,:]

# print(train_encoded_labels.shape)




iterations=100000
learn_rate=0.3
m=pca_images_train.shape[0]
theta=np.zeros((encoded_labels[0].shape[0], pca_images_train.shape[1]))

for itr in range(iterations):
    prediction=np.dot(pca_images_train,theta.T)
    prediction=sigmoid(prediction)
    error=prediction-train_encoded_labels
    theta=theta-(learn_rate * np.dot(error.T,pca_images_train))/m


prediction=np.dot(pca_images_test,theta.T)
prediction=onehotencoder.inverse_transform(prediction)

# print(prediction)

# predict=prediction.argmax(axis=1)
for i in prediction:
	print(dict_names[i[0]])





