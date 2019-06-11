import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.datasets import *
from sklearn.datasets import load_digits
import pandas as pd


def get_Y(Y,classes):
    train_size = len(Y)
    Y_train = np.zeros((train_size,classes))
    for i in range(train_size):
        Y_train[i,Y[i]]=1
    return Y_train

def get_mnist(train_size=1000,test_size=100,classes=10):

    mnist = datasets.fetch_mldata("MNIST Original")

    trX, teX, trY, teY = train_test_split(mnist.data / 255.0, mnist.target.astype("int0"), test_size = 0.0)

    X_train = np.zeros((train_size,trX.shape[1]))
    Y_train = np.zeros((train_size,classes))
    X_test = np.zeros((test_size,trX.shape[1]))
    Y_test = np.zeros((test_size,classes))

    frac = float(1/classes)
    train_per_class = int(X_train.shape[0]*frac)
    test_per_class = int(X_test.shape[0]*frac)

    for i in range(classes):
        locs = np.where(trY==i)
        for j in range(train_per_class):
            X_train[train_per_class*i+j,:]=trX[locs[0][j],:]
            Y_train[train_per_class*i+j,i]=1
        for j in range(test_per_class):
            X_test[test_per_class*i+j,:]=trX[locs[0][train_per_class+j],:]
            Y_test[test_per_class*i+j,i]=1
    
    merged = np.concatenate((X_train,Y_train),axis=1)
    np.random.shuffle(merged)
    X_train = merged[:,:X_train.shape[1]]
    Y_train = merged[:,-Y_train.shape[1]:]

    np.save('./Data/mnist_X_train.npy',X_train)
    np.save('./Data/mnist_X_test.npy',X_test)
    np.save('./Data/mnist_Y_train.npy',Y_train)
    np.save('./Data/mnist_Y_test.npy',Y_test)
    
    return X_train,Y_train,X_test,Y_test

def get_digits(train_size=1000,test_size=100,classes=10):

    (trX,trY) = load_digits(return_X_y=True)

    X_train = np.zeros((train_size,trX.shape[1]))
    Y_train = np.zeros((train_size,classes))
    X_test = np.zeros((test_size,trX.shape[1]))
    Y_test = np.zeros((test_size,classes))

    #per_class_original = len(np.where(trY==0))
    frac = float(1/classes)
    train_per_class = int(X_train.shape[0]*frac//1)
    test_per_class = int(X_test.shape[0]*frac//1)

    for i in range(classes):
        locs = np.where(trY==i)
        #print(len(locs[0]))
        #print(train_per_class)
        for j in range(train_per_class):
            X_train[train_per_class*i+j,:]=trX[locs[0][j],:]
            Y_train[train_per_class*i+j,i]=1
        for j in range(test_per_class):
            X_test[test_per_class*i+j,:]=trX[locs[0][train_per_class+j],:]
            Y_test[test_per_class*i+j,i]=1

    merged = np.concatenate((X_train,Y_train),axis=1)
    np.random.shuffle(merged)
    X_train = merged[:,:X_train.shape[1]]
    Y_train = merged[:,-Y_train.shape[1]:]

    np.save('./Data/digit_X_train.npy',X_train)
    np.save('./Data/digit_X_test.npy',X_test)
    np.save('./Data/digit_Y_train.npy',Y_train)
    np.save('./Data/digit_Y_test.npy',Y_test)
    
    return X_train,Y_train,X_test,Y_test


def get_syn(train_size=1000,test_size=100,classes=10):

    x,trY=make_blobs(n_samples=100000, n_features=classes, centers=classes, shuffle=True)
    dim_1 = classes
    dim_2 = classes*10
    W = np.random.randn(dim_1,dim_2)
    trX = np.matmul(x,W)

    X_train = np.zeros((train_size,trX.shape[1]))
    Y_train = np.zeros((train_size,classes))
    X_test = np.zeros((test_size,trX.shape[1]))
    Y_test = np.zeros((test_size,classes))

    frac = float(1/classes)
    train_per_class = int(X_train.shape[0]*frac)
    test_per_class = int(X_test.shape[0]*frac)
     
    for i in range(classes):
        locs = np.where(trY==i)
        for j in range(train_per_class):
            X_train[train_per_class*i+j,:]=trX[locs[0][j],:]
            Y_train[train_per_class*i+j,i]=1
        for j in range(test_per_class):
            X_test[test_per_class*i+j,:]=trX[locs[0][train_per_class+j],:]
            Y_test[test_per_class*i+j,i]=1

    merged = np.concatenate((X_train,Y_train),axis=1)
    np.random.shuffle(merged)
    X_train = merged[:,:X_train.shape[1]]
    Y_train = merged[:,-Y_train.shape[1]:]

    np.save('./Data/syn_X_train.npy',X_train)
    np.save('./Data/syn_X_test.npy',X_test)
    np.save('./Data/syn_Y_train.npy',Y_train)
    np.save('./Data/syn_Y_test.npy',Y_test)
    
#     u,s,v = np.linalg.svd(W)
#     smat = np.zeros((u.shape[0], v.shape[0]))
#     rank = np.minimum(u.shape[0],v.shape[0])
#     smat[:rank, :rank] = np.diag(1/s)
#     W_inv = np.dot(v.transpose(), np.dot(smat.transpose(), u.transpose()))
    
    return X_train,Y_train,X_test,Y_test

def get_rna(train_size=250,test_size=100):

    data = pd.read_csv('rna_data.csv')
    trX = np.array(data.loc[:,'gene_0':'gene_20530'])
    trY = np.array(data.loc[:,'Class'])

    X_train = np.zeros((train_size,trX.shape[1]))
    Y_train = np.zeros((train_size,np.max(trY)+1))
    X_test = np.zeros((test_size,trX.shape[1]))
    Y_test = np.zeros((test_size,np.max(trY)+1))

    train_per_class = int(X_train.shape[0]*0.2)
    test_per_class = int(X_test.shape[0]*0.2)

    for i in range(5):
        locs = np.where(trY==i)
        for j in range(train_per_class):
            X_train[train_per_class*i+j,:]=trX[locs[0][j],:]
            Y_train[train_per_class*i+j,i]=1
        for j in range(test_per_class):
            X_test[test_per_class*i+j,:]=trX[locs[0][train_per_class+j],:]
            Y_test[test_per_class*i+j,i]=1

    merged = np.concatenate((X_train,Y_train),axis=1)
    np.random.shuffle(merged)
    X_train = merged[:,:X_train.shape[1]]
    Y_train = merged[:,-Y_train.shape[1]:]

    np.save('./Data/rna_X_train.npy',X_train)
    np.save('./Data/rna_X_test.npy',X_test)
    np.save('./Data/rna_Y_train.npy',Y_train)
    np.save('./Data/rna_Y_test.npy',Y_test)
    
    return X_train,Y_train,X_test,Y_test