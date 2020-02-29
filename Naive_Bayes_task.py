# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:43:40 2018

@author: User
"""
#Importing data
import numpy as np
from numpy import loadtxt
data = loadtxt('HW7.txt', delimiter=',')
x=data[:,1:14]
y=data[:,0]

#1. Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

#Splitting the dataset to find the highest score
from sklearn.cross_validation import KFold
kf = KFold(len(x), n_folds=7)
k=0
sm=0
for train_index, test_index in kf:
    #print('k=',k)
    gnb.fit(x[train_index],y[train_index])
    score_test = gnb.score(x[test_index], y[test_index])
    print('score_train=',gnb.score(x[train_index], y[train_index])) 
    print('score_test =',score_test)
    if sm < score_test : 
        #print('k=',k)
        sm=score_test
        train_minindex = train_index
        test_minindex =  test_index
        
    k+=1
    print
gnb.fit(x[train_minindex],y[train_minindex]) 
print(gnb.score(x[test_minindex], y[test_minindex]))


#2. KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
#Splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3)
knn.fit(x_train,y_train)
#Finding train and test score
train_score = knn.score(x_train, y_train)
test_score  = knn.score(x_test, y_test)
print
print('knn','train_score',train_score,'test_score',test_score)
print
#('knn', 'train_score', 0.717741935483871, 'test_score', 0.7592592592592593)
#Printing a confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = knn.predict(x_test)
print '\n confussion matrix for the test set :\n',confusion_matrix(y_test, y_pred)
#Predicting score and confusion matrix for the full dataset
y_p = knn.predict(x)
print
print 'knn.score for the full set :', knn.score(x, y) 
print '\n confussion matrix for the full set :\n',confusion_matrix(y, y_p)
#knn.score for the full set : 0.7303370786516854

#Splitting the data to find the highest score
from sklearn.cross_validation import KFold
kf = KFold(len(x), n_folds=5)
k=0
sm=0
for train_index, test_index in kf:
    #print('k=',k)
    knn.fit(x[train_index],y[train_index])
    score_test = knn.score(x[test_index], y[test_index])
    print('score_train=',knn.score(x[train_index], y[train_index])) 
    print('score_test =',score_test)
    if sm < score_test : 
        #print('k=',k)
        sm=score_test
        train_minindex = train_index
        test_minindex =  test_index
        
    k+=1
    print
#Finding the highest score
knn.fit(x[train_minindex],y[train_minindex])
pred_xtest =knn.predict(x[test_minindex])
print('SCORE on test set: ',knn.score(x[test_minindex],
                                         y[test_minindex]))
#('SCORE on test set: ', 0.8333333333333334)
#Score of the full set
fulldatasc=knn.score(x, y)
print('SCORE on all set: ',knn.score(x,y) )
#('SCORE on all set: ', 0.7415730337078652)


#3. SVC
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
#Fitting a SVM model to the data  
model = SVC(kernel='linear')  
#Splitting the dataset and printing train and test score
svc_x_train, svc_x_test, svc_y_train, svc_y_test= train_test_split(x, y, test_size=0.20)
model.fit(svc_x_train,svc_y_train)
svc_train_score = model.score(svc_x_train, svc_y_train)
svc_test_score  = model.score(svc_x_test, svc_y_test)
print
print(' SVM : ','train_score',svc_train_score,'test_score',svc_test_score)
print
#(' SVM : ', 'train_score', 0.9929577464788732, 'test_score', 0.9722222222222222)
#Creating a confusion matrix
from sklearn.metrics import confusion_matrix
svc_y_pred = model.predict(svc_x_test)
print 'model.score for the test set :', model.score(svc_x_test, svc_y_test) 
print '\n confussion matrixfor for the test set :\n',confusion_matrix(svc_y_test, svc_y_pred)
#model.score for the test set : 0.9444444444444444
#Printing score for the full dataset
svc_y_p = model.predict(x)
print
print 'SVM.score for the full set :',model.score(x, y) 
print '\n confussion matrix for the full set :\n',confusion_matrix(y, svc_y_p)
#SVM.score for the full set : 0.9887640449438202
#Doing a classification report
from sklearn.metrics import classification_report
print(classification_report(svc_y_test,svc_y_pred))

#Splitting the data to find the highest score
from sklearn.cross_validation import KFold
kf = KFold(len(x), n_folds=5)
k=0
sm=0
for train_index, test_index in kf:
    #print('k=',k)
    model.fit(x[train_index],y[train_index])
    score_test = model.score(x[test_index], y[test_index])
    print('score_train=',model.score(x[train_index], y[train_index])) 
    print('score_test =',score_test)
    if sm < score_test :    
        #print('k=',k)
        sm=score_test
        train_minindex = train_index
        test_minindex =  test_index
        
    k+=1
    print
#Finding the highest score
model.fit(x[train_minindex],y[train_minindex])
pred_xtest =model.predict(x[test_minindex])
print('SCORE on test set: ',model.score(x[test_minindex],
                                         y[test_minindex]))
#('SCORE on test set: ', 0.9428571428571428)
#Score of the full set
fulldatasc=model.score(x, y)
print('SCORE on all set: ',model.score(x,y) )
#('SCORE on all set: ', 0.9887640449438202)

