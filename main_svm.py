#!/usr/bin/python

"""
Compare D-sigma, 2-sigma, 1-sigma bolstered resubstitution error estimators.

Feature selection: t-test, isomap
Classification rule: SVM, support vector machine
Error estimators: resubstitution, cross validation with 10 folds, bolstered resubstitution, bootstrap 0.632.
"""

PERCENTILE = 1.0/2
from datetime import datetime
import numpy as np
import sys
from sklearn import svm, cross_validation

def datagen(n, D, d, dt, std):
    """
    Generate dataset for class 1 and class 2

    n -> sample size
    D -> feature size
    d -> marker size
    std -> standard deviation of markers
    output -> two nxD datasets
    """
    n1, n2 = n, n
    m1 = np.zeros(shape=[n1, D])
    m2 = np.zeros(shape=[n2, D])
    std1 = np.ones(shape=[n1, D])
    std2 = np.ones(shape=[n2, D])
    m1[:, 0:d] = -dt# markers with means +-dt
    m2[:, 0:d] = dt
    std1[:, 0:d] = std
    std2[:, 0:d] = std 
    r1 = std1 * np.random.standard_normal(size=[n1, D])
    r2 = std2 * np.random.standard_normal(size=[n1, D])
    X1 = m1 + r1
    X2 = m2 + r2
    return (X1, X2)

def FeaSelTtest(X, y, d):
    """
    Feature selection using t-test

    X -> dataset
    y -> label
    d -> selected feature size
    output -> feature index
    """
    ind1 = y==0
    ind2 = y==1
    X1 = X[ind1]
    X2 = X[ind2]
    m1 = X1.mean(axis=0)
    s1 = X1.std(axis=0)
    m2 = X2.mean(axis=0)
    s2 = X2.std(axis=0)
    t = abs(np.divide((m1-m2),(s1+s2)))#t score  
    ind = np.argsort(t)
    FeaInd = ind[-d:]
    return FeaInd

def FeaSelIsomap(X, y, d):
    """
    Feature selection using isomap idea

    X -> dataset
    y -> label
    d -> selected feature size
    output -> feature index
    """
    n = X.shape[0]
    D = X.shape[1]
    Ds = np.zeros((n, n))#nxn distance matrix
    Dis = np.zeros((n, n, D))#nxnxD distance matrix for each dimension
    for dim in range(D):        
        for i in range(n):
            for j in range(i):
                dij = X[i, dim] -X[j, dim]
                Dis[i, j, dim] = np.sqrt(np.dot(dij, dij))
        for i in range(n):
            for j in range(i, n):
                Dis[i, j, dim] = Dis[j, i, dim]
    for i in range(n):
        for j in range(i):
            dij = X[i, :] -X[j, :]
            Ds[i, j] = np.sqrt(np.dot(dij, dij))
    for i in range(n):
        for j in range(i, n):
            Ds[i, j] = Ds[j, i]
    dists = np.zeros(D)# sum of pairwise distances of each dimension
    distD = np.sum(Ds)# sum of pairwise distances of D dimensions
    for dim in range(D):
        dists[dim] = np.sum(Dis[:, :, dim])
    distDiffs = distD - dists 
    ind = np.argsort(distDiffs)
    FeaInd = ind[-d:]
    return FeaInd

def min_dists_Dsigma(X,y):
    """
    Calculate min distances used for new bolstering sig, D-sigma.

    X -> nxD dataset
    y -> label
    output -> sig1, sig2, for D directions of class 1 and 2.
    """
    from scipy.stats import chi
    ind1 = y==0
    ind2 = y==1
    X1 = X[ind1]
    X2 = X[ind2]
    n = X.shape[0]
    D = X.shape[1]
    X1.sort(axis=0)
    X2.sort(axis=0)
    cp = chi.ppf(PERCENTILE, 1) # degree of one, for every dimention
    sig1 = np.mean(abs(X1[1:]-X1[:-1]),axis=0)/cp
    sig2 = np.mean(abs(X2[1:]-X2[:-1]),axis=0)/cp
    return np.vstack((sig1,sig2))

def min_dists_2sigma(X,y,FeaInd_d):
    """
    Calculate min distances used for new bolstering sig, 2-sigma.

    X -> nxD dataset
    y -> label
    output -> sig11(1xd), sig12(1x(D-d)), sig21(1xd), sig22(1x(D-d)), Four sigmas for two class, with in each class ONE sigma for first d features and the other for the rest of D-d features.
    """
    from scipy.stats import chi
    ind1 = y==0
    ind2 = y==1
    D = X.shape[1]
    FeaInd_D_d = []
    for i in range(D):
        if i not in FeaInd_d:
            FeaInd_D_d.append(i)
    Xd = X[:, FeaInd_d]
    XD_d = X[:, FeaInd_D_d]
    sigs = np.ones((2, D))

    X1d = Xd[ind1]
    X2d = Xd[ind2]
    X1d.sort(axis=0)
    X2d.sort(axis=0)
    cpd = chi.ppf(PERCENTILE, len(FeaInd_d)) # degree of d
    r1d = np.mean(abs(X1d[1:]-X1d[:-1]),axis=0)
    r2d = np.mean(abs(X2d[1:]-X2d[:-1]),axis=0)     
    sig1d = np.sqrt(np.dot(r1d, r1d))/cpd
    sig2d = np.sqrt(np.dot(r2d, r2d))/cpd
    sigs[0, FeaInd_d] = sig1d
    sigs[1, FeaInd_d] = sig2d

    X1D_d = XD_d[ind1]
    X2D_d = XD_d[ind2]
    X1D_d.sort(axis=0)
    X2D_d.sort(axis=0)
    cpD_d = chi.ppf(PERCENTILE, len(FeaInd_D_d)) # degree of D-d
    r1D_d = np.mean(abs(X1D_d[1:]-X1D_d[:-1]),axis=0)
    r2D_d = np.mean(abs(X2D_d[1:]-X2D_d[:-1]),axis=0)
    sig1D_d = np.sqrt(np.dot(r1D_d, r1D_d))/cpD_d
    sig2D_d = np.sqrt(np.dot(r2D_d, r2D_d))/cpD_d
    sigs[0, FeaInd_D_d] = sig1D_d
    sigs[1, FeaInd_D_d] = sig2D_d
    return sigs

def mean_min_dists(X,y):
    """
    Calculate min distances used for original bolstering sig, 1-sigma.

    X -> nxD dataset
    y -> label
    output -> sig11(1xd), sig12(1x(D-d)), sig21(1xd), sig22(1x(D-d)), Four sigmas for two class, with in each class ONE sigma for first d features and the other for the rest of D-d features.
    """

    from scipy.stats import chi
    ind1 = y==0
    ind2 = y==1
    X1 = X[ind1]
    X2 = X[ind2]
    n = X.shape[0]
    p = X.shape[1]
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    tmp1 = np.zeros(n1)
    tmp2 = np.zeros(n2)
    for i in range(n1):
        dm = sys.float_info.max;
        for j in range(i):
            e = X1[i]-X1[j]
            d = np.dot(e,e.T)
            if d<dm:
                dm = d
        for j in range(i+1,n1):
            e = X1[i]-X1[j]
            d = np.dot(e,e.T)            
            if d<dm:
                dm = d
    tmp1[i]=np.sqrt(dm)
    d1 = np.mean(tmp1)
    for i in range(n2):
        dm = sys.float_info.max;
        for j in range(i):
            e = X2[i]-X2[j]
            d = np.dot(e,e.T)            
            if d<dm:
                dm = d
        for j in range(i+1,n2):
            e = X2[i]-X2[j]
            d = np.dot(e,e.T)            
            if d<dm:
                dm = d
    tmp2[i]=np.sqrt(dm)
    d2 = np.mean(tmp2)

    cp = chi.ppf(PERCENTILE, p)
    sig1 = d1*np.ones(p)/cp
    sig2 = d2*np.ones(p)/cp
    return np.vstack((sig1,sig2))

def crossValidation(X_train, y_train, d, nFolds):
    """
    svm cross validation (need feature selection for each XD_train)

    X_train -> nxD
    y_train -> n
    d -> selected feature size
    nFolds -> by default, 10 folds.
    output -> error rate
    """
    kf = cross_validation.KFold(len(y_train),n_folds=nFolds,indices=True)
    errs = []
    for train_index, test_index in kf:
        Xcv10_train = X_train[train_index]
        Xcv10_test = X_train[test_index]
        ycv10_train = y_train[train_index]
        ycv10_test = y_train[test_index]
        FeaInd_cv10 = FeaSelTtest(Xcv10_train,ycv10_train,d)    
        Xdcv10_train = Xcv10_train[:,FeaInd_cv10]
        Xdcv10_test = Xcv10_test[:,FeaInd_cv10]
        clf = svm.SVC(kernel='linear')
        clf.fit(Xdcv10_train,ycv10_train)
        err = 1-clf.score(Xdcv10_test, ycv10_test)    
        errs.append(err)
    err_cv10 = np.array(errs).mean()
    return err_cv10

def bootstrap(X_train, y_train, d, nIter=100, random_state=0):
    """
    svm bootstrap 0 and 632.

    X_train -> nxD
    y_train -> n
    d -> selected feature size
    output -> err_bs0, err_bs632
    """
    bs = cross_validation.Bootstrap(len(y_train),n_iter=nIter,random_state=random_state)
    errs0 = []
    errs632 = []
    for train_index, test_index in bs:
        Xbs_train = X_train[train_index]
        Xbs_test = X_train[test_index]
        ybs_train = y_train[train_index]
        ybs_test = y_train[test_index]
        FeaInd_bs = FeaSelTtest(Xbs_train,ybs_train,d)    
        Xdbs_train = Xbs_train[:,FeaInd_bs]
        Xdbs_test = Xbs_test[:,FeaInd_bs]
        clf = svm.SVC(kernel='linear')
        clf.fit(Xdbs_train,ybs_train)
        err0 = 1-clf.score(Xdbs_test, ybs_test)
        err_bs_resub = 1-clf.score(Xdbs_train, ybs_train)
        err632 = (1-0.632)*err_bs_resub+0.632*err0
        errs0.append(err0)
        errs632.append(err632)
    err_bs0 = np.array(errs0).mean()
    err_bs632 = np.array(errs632).mean()
    return (err_bs0, err_bs632)

def brNewDsig(X_train, y_train, Xd_train, yd_train, FeaInd_d, MC=10):
    """
    svm bolstered resubstitution, D-sigma
    
    X_train -> nxD
    y_train -> n
    Xd_train -> nxd
    yd_train -> n
    FeaInd_d -> d feature indices
    MC -> Monte Carlo numbers, by default 10
    output -> error rate
    """
    clf = svm.SVC(kernel='linear')
    clf.fit(Xd_train,yd_train)    
    sigs = min_dists_Dsigma(X_train,y_train)
    sigs = sigs*np.sqrt(len(FeaInd_d))
    sigs = sigs*np.std(X_train[FeaInd_d], axis=0)

    ind1 = y_train==0
    ind2 = y_train==1
    X1_train = X_train[ind1]
    X2_train = X_train[ind2]
    y1_train = y_train[ind1]
    y2_train = y_train[ind2]
    MC = MC
    errs = []
    for mc in range(MC):
        r1 = np.random.multivariate_normal\
            (np.zeros(X_train.shape[1]),np.diag(sigs[0]),\
                 X1_train.shape[0])
        X1_br_train = X1_train + r1
        r2 = np.random.multivariate_normal\
            (np.zeros(X_train.shape[1]),np.diag(sigs[1]),\
                 X2_train.shape[0])
        X2_br_train = X2_train + r2
        Xd1_br_train = X1_br_train[:,FeaInd_d]
        Xd2_br_train = X2_br_train[:,FeaInd_d]
        Xd_br_train = np.vstack((Xd1_br_train,Xd2_br_train))
        yd_br_train = np.concatenate((y1_train,y2_train))
        err = 1-clf.score(Xd_br_train,yd_br_train)
        errs.append(err)
    err_brNEWD = np.array(errs).mean()
    return err_brNEWD

def brNew2sig(X_train, y_train, Xd_train, yd_train, FeaInd_d, MC=10):
    """
    svm bolstered resubstitution, 2-sigma
    
    X_train -> nxD
    y_train -> n
    Xd_train -> nxd
    yd_train -> n
    FeaInd_d -> d feature indices
    MC -> Monte Carlo numbers, by default 10
    output -> error rate
    """
    clf = svm.SVC(kernel='linear')
    clf.fit(Xd_train,yd_train)
    sigs = min_dists_2sigma(X_train,y_train, FeaInd_d)
    sigs = sigs*np.sqrt(len(FeaInd_d))
    sigs = sigs*np.std(X_train[FeaInd_d], axis=0)

    ind1 = y_train==0
    ind2 = y_train==1
    X1_train = X_train[ind1]
    X2_train = X_train[ind2]
    y1_train = y_train[ind1]
    y2_train = y_train[ind2]
    MC = MC
    errs = []
    for mc in range(MC):
        r1 = np.random.multivariate_normal\
            (np.zeros(X_train.shape[1]),np.diag(sigs[0]),\
                 X1_train.shape[0])
        X1_br_train = X1_train + r1
        r2 = np.random.multivariate_normal\
            (np.zeros(X_train.shape[1]),np.diag(sigs[1]),\
                 X2_train.shape[0])
        X2_br_train = X2_train + r2
        Xd1_br_train = X1_br_train[:,FeaInd_d]
        Xd2_br_train = X2_br_train[:,FeaInd_d]
        Xd_br_train = np.vstack((Xd1_br_train,Xd2_br_train))
        yd_br_train = np.concatenate((y1_train,y2_train))
        err = 1-clf.score(Xd_br_train,yd_br_train)
        errs.append(err)
    err_brNEWD = np.array(errs).mean()
    return err_brNEWD

def brOLD(X_train, y_train, Xd_train, yd_train, FeaInd_d, MC=10):
    """
    svm bolstered resubstitution, original 1-sigma
    
    X_train -> nxD
    y_train -> n
    Xd_train -> nxd
    yd_train -> n
    FeaInd_d -> d feature indices
    MC -> Monte Carlo numbers, by default 10
    output -> error rate
    """
    clf = svm.SVC(kernel='linear')
    clf.fit(Xd_train,yd_train)
    sigs = mean_min_dists(X_train,y_train)
    ind1 = y_train==0
    ind2 = y_train==1
    X1_train = X_train[ind1]
    X2_train = X_train[ind2]
    y1_train = y_train[ind1]
    y2_train = y_train[ind2]
    MC = MC
    errs = []
    for mc in range(MC):
        r1 = np.random.multivariate_normal\
            (np.zeros(X_train.shape[1]),np.diag(sigs[0]),\
                 X1_train.shape[0])
        X1_br_train = X1_train + r1
        r2 = np.random.multivariate_normal\
            (np.zeros(X_train.shape[1]),np.diag(sigs[1]),\
                 X2_train.shape[0])
        X2_br_train = X2_train + r2
        Xd1_br_train = X1_br_train[:,FeaInd_d]
        Xd2_br_train = X2_br_train[:,FeaInd_d]
        Xd_br_train = np.vstack((Xd1_br_train,Xd2_br_train))
        yd_br_train = np.concatenate((y1_train,y2_train))
        err = 1-clf.score(Xd_br_train,yd_br_train)
        errs.append(err)
    err_brOLD = np.array(errs).mean()
    return err_brOLD

def main():
    ds = [5,10,15,20]
    tstart = datetime.now()
    n = 300
    D = 100
    d0 = 15
    dt = 0.3
    stds= [0.5,1.0,2.0]
    N = 200
    for std in stds:
        for d in ds:
            fn = "svm_dt%d_ttest_d%d_std_%.1f.txt" % ((dt*10),d,std)
            f = open(fn,'w')
            f.write("true\tresub\tcv10\tloo\tbs0\tbs632\tbrNEWD\tbrOLD\tbs_br_mixed\tbrNEW2sig\n")
            for i in range(N):
                X1, X2 = datagen(n,D,d0,dt,std)
                y1 = np.zeros((X1.shape[0]))
                y2 = np.ones((X2.shape[0]))
                XD = np.vstack((X1,X2))
                y = np.concatenate((y1,y2)) 
                FeaInd_d = FeaSelTtest(XD,y,d)
                # randomly split data into train and test
                X_train, X_test, y_train, y_test = \
                         cross_validation.train_test_split\
                         (XD, y,test_size=0.80)
                Xd_train, Xd_test, yd_train, yd_test = \
                          X_train[:,FeaInd_d], X_test[:,FeaInd_d],\
                          y_train, y_test        
                # svm true error
                clf = svm.SVC(kernel='linear')
                clf.fit(Xd_train,yd_train)
                err_true = 1-clf.score(Xd_test, yd_test)
                # svm resubsitution
                clf = svm.SVC(kernel='linear')
                clf.fit(Xd_train,yd_train)
                err_resub = 1-clf.score(Xd_train,yd_train)
                # svm cv10
                err_cv10 = crossValidation(X_train, y_train, d, 10)
                # svm loo
                err_loo = crossValidation(X_train, y_train, d, len(y_train))
                # svm bootstrap 0 and 632
                err_bs0, err_bs632 = bootstrap(X_train, y_train, d, nIter=100, random_state=0)
                # svm bolstered resubstitution NEW D=100
                err_brNEWD = brNewDsig(X_train, y_train, Xd_train, yd_train, FeaInd_d, MC=10)
                err_brOLD = brOLD(X_train, y_train, Xd_train, yd_train, FeaInd_d, MC=10)
                err_bs632_brNEWD = 0.5*err_bs632+0.5*err_brNEWD
                err_brNEW2sig = brNew2sig(X_train, y_train, Xd_train, yd_train, FeaInd_d, MC=10)
                f.write("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" \
                        % (err_true,err_resub,err_cv10,err_loo,
                           err_bs0,err_bs632,err_brNEWD,err_brOLD,err_bs632_brNEWD, err_brNEW2sig))
            f.close()
            print 'innerloop'
    print "Done"
    tend = datetime.now()
    print tend - tstart
if __name__ == '__main__':
    print __doc__
    main()
