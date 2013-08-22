#!/usr/bin/python

"""
Test feature selection methods, sigma calculating methods.

"""

import numpy as np
import sys
from main_svm import *

def test_FeaSel():
    n = 60
    D = 100
    d0 = 15
    dt = 0.3
    std = 0.5#1.0,2.0
    ds = [5,10,15,20]

    X1, X2 = datagen(n,D,d0,dt,std)
    y1 = np.zeros((X1.shape[0]))
    y2 = np.ones((X2.shape[0]))
    XD = np.vstack((X1,X2))
    y = np.concatenate((y1,y2)) 
    for d in ds:
        FeaInd_d = FeaSelTtest(XD, y, d)
        FeaInd_d2 = FeaSelIsomap(XD, y, d)
        print FeaInd_d
        print FeaInd_d2


def test_dists():
    n = 60
    D = 10
    d0 = 15
    dt = 0.3
    std = 1.0#0.5#1.0#1.0,2.0
    ds = [5,10]#,15,20]

    X1, X2 = datagen(n,D,d0,dt,std)
    y1 = np.zeros((X1.shape[0]))
    y2 = np.ones((X2.shape[0]))
    XD = np.vstack((X1,X2))
    y = np.concatenate((y1,y2)) 

    for d in ds:
        FeaInd_d = FeaSelTtest(XD,y,d)
        FeaInd_d2 = FeaSelIsomap(XD,y,d)
        print FeaInd_d
        print FeaInd_d2
        X_train, X_test, y_train, y_test = \
                         cross_validation.train_test_split\
                         (XD, y,test_size=0.80)
        sigs1 = min_dists_Dsigma(X_train, y_train)
        sigs2 = min_dists_2sigma(X_train, y_train, FeaInd_d)
        sigs3 = mean_min_dists(X_train,y_train)
        np.set_printoptions(precision=3)
        print sigs1*np.sqrt(d)
        print sigs2*np.sqrt(d)
        print sigs3*np.sqrt(d)
def main():
    # test_FeaSel()
    test_dists()
if __name__ == '__main__':
    print __doc__
    main()
