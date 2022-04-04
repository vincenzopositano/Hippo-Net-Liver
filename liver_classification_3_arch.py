# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 15:15:38 2021

@author: ippop
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')


import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
 
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


 
 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

from pydicom import dcmread
 
import os

tf. device("cpu:0")



data_dir =r'C:\vincenzo\MIOT\articolo DL Fegato\CODE'
img_height , img_width = 64, 64
seq_len = 10
 
#classes = ["Borderline", "Mild", "Moderate", "Normal", "Severe"]
classes = ["Normal","Borderline", "Mild", "Moderate","Severe"]

#load data

X=np.load(data_dir+'\X_MultiScanner.npy')
Y=np.load(data_dir+'\Y_MultiScanner.npy')


# print data distribution
print('Normal = ',np.sum(Y[:,0]))
print('Borderline = ',np.sum(Y[:,1]))
print('Mild = ',np.sum(Y[:,2]))
print('Moderate = ',np.sum(Y[:,3]))
print('Severe = ',np.sum(Y[:,4]))

rstate=88 # random state
file_LSTM='model_current_LSTM_'+str(rstate)
file_2D='model_current_2D_'+str(rstate)
file_3D='model_current_3D_'+str(rstate)

#create test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True,random_state=rstate,stratify=Y)
 
# print test set distribution
print('Test set distribution')
print('Normal = ',np.sum(y_test[:,0]))
print('Borderline = ',np.sum(y_test[:,1]))
print('Mild = ',np.sum(y_test[:,2]))
print('Moderate = ',np.sum(y_test[:,3]))
print('Severe = ',np.sum(y_test[:,4]))


from timeit import default_timer as timer
from datetime import timedelta

# ground truth
y_test1 = np.argmax(y_test, axis = 1) 
y_test_label_GT=[None] * np.size(y_test1)
for c in range(len(classes)):
    id=np.array(np.where(y_test1==c))
    for i in range (np.size(id)):
        y_test_label_GT[id[0,i]]=classes[c]
        

# load LSTM model
model=tf.keras.models.load_model(file_LSTM)
batchSize=32

start = timer()

y_pred_LTSM = model.predict(X_test,batch_size=batchSize)

end = timer()
print('Classification  Time LSTM = ',timedelta(seconds=end-start))


#analyze results

y_test1 = np.argmax(y_pred_LTSM, axis = 1) 
y_test_label_LSTM=[None] * np.size(y_test1)
for c in range(len(classes)):
    id=np.array(np.where(y_test1==c))
    for i in range (np.size(id)):
        y_test_label_LSTM[id[0,i]]=classes[c]
        

# load 2D model
model=tf.keras.models.load_model(file_2D)
batchSize=32

s=np.shape(X_test)
X_test_2D=np.zeros([s[0],s[2],s[3],s[1]])
for p in range(s[0]):
    for i in range(s[1]):
        X_test_2D[p,:,:,i]=X_test[p,i,:,:,0]


start = timer()

y_pred_2D = model.predict(X_test_2D,batch_size=batchSize)

end = timer()
print('Classification  Time 2D = ',timedelta(seconds=end-start))


#analyze results

y_test1 = np.argmax(y_pred_2D, axis = 1) 
y_test_label_2D=[None] * np.size(y_test1)
for c in range(len(classes)):
    id=np.array(np.where(y_test1==c))
    for i in range (np.size(id)):
        y_test_label_2D[id[0,i]]=classes[c]      

# load 3D model
model=tf.keras.models.load_model(file_3D)
batchSize=32

start = timer()

y_pred_3D = model.predict(X_test,batch_size=batchSize)

end = timer()
print('Classification  Time 3D = ',timedelta(seconds=end-start))


#analyze results

y_test1 = np.argmax(y_pred_3D, axis = 1) 
y_test_label_3D=[None] * np.size(y_test1)
for c in range(len(classes)):
    id=np.array(np.where(y_test1==c))
    for i in range (np.size(id)):
        y_test_label_3D[id[0,i]]=classes[c]        


print(' 2D Network performences')   
print('Accuracy test set 2D = ',accuracy_score(y_test_label_GT,y_test_label_2D))
print(' 2D Class Data')   
confusionMatrix=confusion_matrix(y_test_label_GT,y_test_label_2D,labels=classes,normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,display_labels=classes)
disp.plot()
FP = confusionMatrix.sum(axis=0) - np.diag(confusionMatrix) 
FN = confusionMatrix.sum(axis=1) - np.diag(confusionMatrix)
TP = np.diag(confusionMatrix)
TN = confusionMatrix.sum() - (FP + FN + TP)

confusionMatrix2D=confusionMatrix
np.save('confusionMatrix2D'+str(rstate),confusionMatrix2D)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)

print(TNR)
print(TPR)
print(ACC)

print(' 3D Network performences')  
print('Accuracy test set 3D = ',accuracy_score(y_test_label_GT,y_test_label_3D))
print('3D  Class Data')   
confusionMatrix=confusion_matrix(y_test_label_GT,y_test_label_3D,labels=classes,normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,display_labels=classes)
disp.plot()
FP = confusionMatrix.sum(axis=0) - np.diag(confusionMatrix) 
FN = confusionMatrix.sum(axis=1) - np.diag(confusionMatrix)
TP = np.diag(confusionMatrix)
TN = confusionMatrix.sum() - (FP + FN + TP)

confusionMatrix3D=confusionMatrix
np.save('confusionMatrix3D'+str(rstate),confusionMatrix3D)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)

print(TNR)
print(TPR)
print(ACC)


print(' LSTM Network performences')  
print('Accuracy test set LSTM = ',accuracy_score(y_test_label_GT,y_test_label_LSTM))
print('LSTM Class Data')   
confusionMatrix=confusion_matrix(y_test_label_GT,y_test_label_LSTM,labels=classes,normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,display_labels=classes)
disp.plot()
FP = confusionMatrix.sum(axis=0) - np.diag(confusionMatrix) 
FN = confusionMatrix.sum(axis=1) - np.diag(confusionMatrix)
TP = np.diag(confusionMatrix)
TN = confusionMatrix.sum() - (FP + FN + TP)

confusionMatrixLSTM=confusionMatrix
np.save('confusionMatrixLSTM'+str(rstate),confusionMatrixLSTM)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)

print(TNR)
print(TPR)
print(ACC)

from collections import Counter
y_test_label_comb=[None]*np.size(y_test1)
y_comb=[y_test_label_3D,y_test_label_2D,y_test_label_LSTM] # combine lists
for i in range (len(y_test_label_2D)):
    tmp=[y_comb[0][i],y_comb[1][i],y_comb[2][i]]
    c=Counter(tmp)
    mc=c.most_common(1)
    y_test_label_comb[i]=mc[0][0]


y_comb1=[y_test_label_2D,y_test_label_3D,y_test_label_LSTM,y_test_label_comb] # combine lists

print('Accuracy test set COMB = ',accuracy_score(y_test_label_GT,y_test_label_comb))

     

print(classification_report(y_test_label_GT, y_test_label_comb,labels=classes))



confusionMatrix=confusion_matrix(y_test_label_GT,y_test_label_comb,labels=classes,normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,display_labels=classes)
disp.plot()


confusionMatrixENSEMBLE=confusionMatrix
np.save('confusionMatrixENSEMBLE'+str(rstate),confusionMatrixENSEMBLE)


FP = confusionMatrix.sum(axis=0) - np.diag(confusionMatrix) 
FN = confusionMatrix.sum(axis=1) - np.diag(confusionMatrix)
TP = np.diag(confusionMatrix)
TN = confusionMatrix.sum() - (FP + FN + TP)



# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)



print(TNR)
print(TPR)
print(ACC)
