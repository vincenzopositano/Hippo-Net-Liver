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
import sys
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
import matplotlib.pyplot as plt
 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"



data_dir ='/home/positano/Desktop/DEEP_LEARNING/DL_DATA'
img_height , img_width = 64, 64
seq_len = 10
 
#classes = ["Borderline", "Mild", "Moderate", "Normal", "Severe"]
classes = ["Normal","Borderline", "Mild", "Moderate","Severe"]

# image crop function
def frames_crop(frames,size):
    
    import skfuzzy as fuzz 
    
    image=np.asarray(frames)
    image1=frames[0]
    
    imgdata=np.reshape(image,(len(frames),image1.shape[0]*image1.shape[1]))
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(imgdata, 3, 2, error=0.005, maxiter=1000, init=None)
    id0 =np.argsort(cntr[:,0],axis=0)
    backMap=np.reshape(u[id0[0],:],(image1.shape[0],image1.shape[1]))
    bodyMap=np.uint8((1-backMap)>0.5)
    

    
    num_labels, labels_im = cv2.connectedComponents(bodyMap)
    mask = np.zeros(bodyMap.shape, dtype="uint8")
    area=np.zeros(num_labels)
    for i in range(1, num_labels):
        area[i] = np.sum(np.uint8(labels_im==i))
    idmax =np.flip(np.argsort(area))
    bodyMap=np.uint8(labels_im==idmax[0])
    for i in range(1,num_labels):
        if (area[idmax[i]]>2000):
            bodyMap=bodyMap+np.uint8(labels_im==idmax[i])
    

    coords=np.column_stack(np.where(bodyMap==1))

    min_x=np.amin(coords[:,0])
    max_x=np.amax(coords[:,0])
    min_y=np.amin(coords[:,1])
    max_y=np.amax(coords[:,1])

    crop=image1[min_x:max_x,min_y:max_y]
    crop=crop[:,0:np.int32(np.ceil(crop.shape[1]/2))]
    crop=cv2.resize(crop,size)
    
    # import matplotlib.pyplot as plt
    # import matplotlib.image as mpimg
    # imgplot = plt.imshow(crop,cmap='gray')
    # plt.show()
    


    
    crop=np.expand_dims(crop,axis=2)
    
    cropped_frames=[]
    cropped_frames.append(crop)
    
    for i in range(1, len(frames)):
        image=frames[i]
        crop=image[min_x:max_x,min_y:max_y]
        crop=crop[:,0:np.int32(np.ceil(crop.shape[1]/2))]
        crop=cv2.resize(crop,size)
        # imgplot = plt.imshow(crop,cmap='gray')
        # plt.show()
        crop=np.expand_dims(crop,axis=2)
        cropped_frames.append(crop)
    
    
    
     
    return cropped_frames
 
#  Creating frames from videos
 
def frames_extraction(video_path):
    frames_list = []
     
    vidObj = cv2.VideoCapture(video_path)
    # Used as counter variable 
    count = 1
 
    while count <= seq_len: 
         
        success, image = vidObj.read() 
        if success:
            
            image = cv2.resize(image, (img_height, img_width))
            frames_list.append(image)
            count += 1
        else:
            print("Defected frame")
            break
 
            
    return frames_list
 
def create_data(input_dir):
    X = []
    Y = []
     
    classes_list = os.listdir(input_dir)  # classes directories 
    # print(classes_list)
     
    for c in classes_list:
        print(c)
        files_list = os.listdir(os.path.join(input_dir, c)) # patient's directories
        for f in files_list:
            image_list = os.listdir(os.path.join(os.path.join(input_dir, c), f)) # image files
            frames = []
            count = 0
            print(f)
            while count < seq_len: 
                dcm_file=os.path.join(os.path.join(os.path.join(input_dir, c), f),image_list[count])
                dcm_img=dcmread(dcm_file)
                #print(dcm_file)
                #image = cv2.resize(dcm_img.pixel_array, (img_height, img_width))
                image=dcm_img.pixel_array
                frames.append(image)
                count += 1
                
            cropped_frames=frames_crop(frames,(img_height, img_width))
            
            
            X.append(cropped_frames)
             
            y = [0]*len(classes)
            y[classes.index(c)] = 1
            Y.append(y)
     
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y


def normalizeData(Xor):
    X = []
    dataSize=Xor.shape[0]
    seqLen=Xor.shape[1]
    for i in range(dataSize):
        orXData=Xor[i,:,:,:,0] # append original data
        orXData=orXData*255.0/np.amax(orXData)
        X.append(orXData)
        
    X = np.asarray(X)
    X=np.expand_dims(X,axis=4)
    return X
    

from IPython.display import clear_output

class PlotLearning(tf.keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
            

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        # Plotting
        metrics = [x for x in logs if 'val' not in x]
        
        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2), 
                        self.metrics[metric], 
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2), 
                            self.metrics['val_' + metric], 
                            label='val_' + metric)
                
            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()







def createAugmentedData(Xor,Yor,inc):
    X = []
    Y = []
    dataSize=Xor.shape[0]
    seqLen=Xor.shape[1]
    rng = np.random.default_rng(seed=42)
    for i in range(dataSize):
        print(i)
        orXData=Xor[i,:,:,:,0] # append original data
        orYData=Yor[i,:]
        X.append(orXData)
        Y.append(orYData)
        for j in range(inc):
            CurrentImg=Xor[i,:,:,:,0] #first frame
            augImg=CurrentImg
            augImg=tf.keras.preprocessing.image.random_rotation(augImg,20,channel_axis=0,fill_mode='nearest')
            augImg=tf.keras.preprocessing.image.random_zoom(augImg,[1.0,1.6],channel_axis=0,fill_mode='nearest')
            augImg=tf.keras.preprocessing.image.random_shift(augImg,0.2,0.2,channel_axis=0,fill_mode='nearest')
            rfloat = 1+rng.random()
            augImg=augImg*rfloat
            rfloat = 50*rng.random()
            augImg=augImg+rfloat
            
            X.append(augImg)
            orYData=Yor[i,:]
            Y.append(orYData)
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    X=np.expand_dims(X,axis=4)
    
    return X,Y
    
def createAugmentedData1(Xor,Yor,inc):
    X = []
    Y = []
    dataSize=Xor.shape[0]
    seqLen=Xor.shape[1]
    rng = np.random.default_rng(seed=42)
    for i in range(dataSize):
        orXData=Xor[i,:,:,:,0] # append original data
        orYData=Yor[i,:]
        X.append(orXData)
        Y.append(orYData)
        for j in range(inc):
            CurrentImg=Xor[i,:,:,:,0] # original data
            augImg=CurrentImg
            p=rng.integers(1,6)
            
            if (p ==1):
                    augImg=tf.keras.preprocessing.image.random_rotation(augImg,30,channel_axis=0,fill_mode='nearest')
            if (p ==2):
                #augImg=tf.keras.preprocessing.image.random_zoom(augImg,[1.0,1.6],channel_axis=0,fill_mode='nearest')
                augImg=tf.keras.preprocessing.image.random_shear(augImg,40,channel_axis=0,fill_mode='nearest')
            if (p ==3):
                    augImg=tf.keras.preprocessing.image.random_shift(augImg,0.3,0.3,channel_axis=0,fill_mode='nearest')
            if (p ==4):
                    rfloat = 1+rng.random()
                    augImg=augImg*rfloat
            if (p ==5):
                    rfloat = 100*rng.random()
                    augImg=augImg+rfloat
            
            X.append(augImg)
            orYData=Yor[i,:]
            Y.append(orYData)
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    X=np.expand_dims(X,axis=4)
    
    return X,Y   

#X, Y = create_data(data_dir)
#sys.exit()
#X=X*np.amax(X)/255

X=np.load('/home/positano/Desktop/DEEP_LEARNING/X_1069_zoom.npy')
Y=np.load('/home/positano/Desktop/DEEP_LEARNING/Y_1069_zoom.npy')

#X=X*np.amax(X)/255

# X1=np.zeros((X.shape[0],2*X.shape[1],X.shape[2],X.shape[3],1),dtype=np.int16)
# X1[:,0:X.shape[1],:,:,:]=X
# X1[:,X.shape[1]:2*X.shape[1],:,:,:]=X
# X=X1
# seq_len=2

print('Normal = ',np.sum(Y[:,0]))
print('Borderline = ',np.sum(Y[:,1]))
print('Mild = ',np.sum(Y[:,2]))
print('Moderate = ',np.sum(Y[:,3]))
print('Severe = ',np.sum(Y[:,4]))


Normal=np.sum(Y[:,0])
Borderline = np.sum(Y[:,1])
Mild = np.sum(Y[:,2])
Moderate = np.sum(Y[:,3])
Severe = np.sum(Y[:,4])
Total=Normal+Borderline+Mild++Moderate+Severe
cw1=[Normal,Borderline,Mild,Moderate,Severe]
cw=(Total/cw1)/np.sum(Total/cw1)
classW={0:cw[0],1:cw[1],2:cw[2],3:cw[3],4:cw[4]}

rstate=88

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True,random_state=rstate,stratify=Y)
 
print('Train set distribution')
print('Normal = ',np.sum(y_train[:,0]))
print('Borderline = ',np.sum(y_train[:,1]))
print('Mild = ',np.sum(y_train[:,2]))
print('Moderate = ',np.sum(y_train[:,3]))
print('Severe = ',np.sum(y_train[:,4]))
print('Test set distribution')
print('Normal = ',np.sum(y_test[:,0]))
print('Borderline = ',np.sum(y_test[:,1]))
print('Mild = ',np.sum(y_test[:,2]))
print('Moderate = ',np.sum(y_test[:,3]))
print('Severe = ',np.sum(y_test[:,4]))


model = Sequential()
kSize=3
fSize=10
bSize=32
regValue=0.005

model.add(Conv3D(filters = bSize, kernel_size = (fSize, kSize,kSize), padding='same', activation='relu',kernel_initializer='he_normal',input_shape = (seq_len, img_height, img_width,1)))
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(1, 2, 2)))

model.add(Conv3D(filters = bSize, kernel_size = (fSize, kSize,kSize), padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l=regValue)))
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(1, 2, 2)))

model.add(Conv3D(filters = bSize, kernel_size = (fSize, kSize,kSize), padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l=regValue)))
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(1, 2, 2)))

model.add(Conv3D(filters = bSize, kernel_size = (fSize, kSize,kSize), padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l=regValue)))
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(1, 2, 2)))

model.add(Conv3D(filters = bSize, kernel_size = (fSize, kSize,kSize), padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l=regValue)))
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(1, 2, 2)))

model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(5, activation = "softmax"))

# model.add(ConvLSTM2D(filters = 64, kernel_size = (5, 5), padding='valid',return_sequences = False, data_format = "channels_last", input_shape = (seq_len, img_height, img_width,1)))
# model.add(BatchNormalization())
# model.add(Conv2D(filters=64, kernel_size= (3,3), activation="relu"))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(filters=64, kernel_size= (3,3), activation="relu"))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# # model.add(Dense(, activation="relu"))
# # model.add(BatchNormalization())
# # model.add(Dropout(0.3))
# model.add(Dense(5, activation = "softmax"))

#model.add(ConvLSTM2D(filters = 64, kernel_size = (3, 3), padding='valid',return_sequences = False, data_format = "channels_last", input_shape = (seq_len, img_height, img_width,1)))
# model.add(BatchNormalization())
# model.add(Dropout(0.2)) # avoid overfitting
# model.add(BatchNormalization())
# model.add(Dense(512, activation="relu"))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))
# model.add(Dense(5, activation = "softmax"))
 
model.summary()
 

def step_decay(epoch):
   initial_lrate = 0.001
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * np.power(drop,np.floor((1+epoch)/epochs_drop))
   return lrate

lrate = LearningRateScheduler(step_decay)



opt = tf.keras.optimizers.Adam(learning_rate=0.0001,amsgrad=True)
opt1=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.7, decay=0.01, nesterov=False)
model.compile(loss='kullback_leibler_divergence', optimizer=opt, metrics=["accuracy"])
#model.compile(loss='categorical_hinge', optimizer=opt, metrics=["accuracy"])
 
earlystop = EarlyStopping(monitor='val_accuracy',patience=100,restore_best_weights=True)
callbacks = [earlystop, PlotLearning()]

epochs=500
batchSize=32

# train_datagen = ImageDataGenerator(rotation_range=5,  # rotation
#                                    width_shift_range=0.2,  # horizontal shift
#                                    zoom_range=0.2,  # zoom
#                                    horizontal_flip=True,  # horizontal flip
#                                    brightness_range=[0.2,0.8])  # brightness


# from tweaked_ImageGenerator_v2 import ImageDataGenerator
# train_datagen = ImageDataGenerator()
# train_data=train_datagen.flow(X_train,y_train,batch_size=64, seed=27,shuffle=False, frames_per_step=10)

#X_train,y_train = createAugmentedData1(X_train,y_train,20)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, shuffle=True,random_state=rstate)
X_train,y_train = createAugmentedData1(X_train,y_train,20)
#X_train=normalizeData(X_train)
print('augmentation done')

from timeit import default_timer as timer
from datetime import timedelta

start = timer()

history = model.fit(x = X_train, y= y_train, epochs=epochs,batch_size = batchSize , class_weight=classW,steps_per_epoch=50,shuffle=True, validation_data=(X_val,y_val),callbacks=callbacks)


end = timer()
print('Fitting Time = ',timedelta(seconds=end-start))


model.save('/home/positano/Desktop/DEEP_LEARNING/model_current_3D_88')



loss =  history.history['loss']
accuracy = history.history['accuracy']

val_loss =  history.history['val_loss']
val_accuracy = history.history['val_accuracy']

epochs_range = range(len(history.history['loss']))


plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, accuracy, label='Trainig Accuracy')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.legend(loc='upper right')
plt.title('Training Loss and Accuracy')
plt.ylim([0, 1])
plt.show()



y_pred = model.predict(X_val,batch_size=batchSize)

y_pred1 = np.argmax(y_pred, axis = 1)
y_val1 = np.argmax(y_val, axis = 1)

y_pred_label=[None] * np.size(y_pred1)
for c in range(len(classes)):
    id=np.array(np.where(y_pred1==c))
    for i in range (np.size(id)):
        y_pred_label[id[0,i]]=classes[c]
        
y_val_label=[None] * np.size(y_pred1)
for c in range(len(classes)):
    id=np.array(np.where(y_val1==c))
    for i in range (np.size(id)):
        y_val_label[id[0,i]]=classes[c]

print(classification_report(y_val_label, y_pred_label,labels=classes))

# ConfusionMatrix=confusion_matrix(y_test_label,y_pred_label,labels=classes)
# ConfusionMatrixDisplay.from_predictions(y_test_label, y_pred_label,labels=classes)


print('Accuracy = ',accuracy_score(y_val_label,y_pred_label))

confusionMatrix=confusion_matrix(y_val_label,y_pred_label,labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,display_labels=classes)
disp.plot()

confusionMatrix=confusion_matrix(y_val_label,y_pred_label,labels=classes,normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,display_labels=classes)
disp.plot()


#X_test=normalizeData(X_test)
y_pred = model.predict(X_test,batch_size=batchSize)

y_pred1 = np.argmax(y_pred, axis = 1)
y_test1 = np.argmax(y_test, axis = 1)

y_pred_label=[None] * np.size(y_pred1)
for c in range(len(classes)):
    id=np.array(np.where(y_pred1==c))
    for i in range (np.size(id)):
        y_pred_label[id[0,i]]=classes[c]
        
y_test_label=[None] * np.size(y_pred1)
for c in range(len(classes)):
    id=np.array(np.where(y_test1==c))
    for i in range (np.size(id)):
        y_test_label[id[0,i]]=classes[c]

print(classification_report(y_test_label, y_pred_label,labels=classes))

# ConfusionMatrix=confusion_matrix(y_test_label,y_pred_label,labels=classes)
# ConfusionMatrixDisplay.from_predictions(y_test_label, y_pred_label,labels=classes)


print('Accuracy = ',accuracy_score(y_test_label,y_pred_label))

confusionMatrix=confusion_matrix(y_test_label,y_pred_label,labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,display_labels=classes)
disp.plot()

confusionMatrixN=confusion_matrix(y_test_label,y_pred_label,labels=classes,normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrixN,display_labels=classes)
disp.plot()



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

