"""""
lstm - movies
"""""
##############  import  #############
from tensorflow.keras import optimizers
from tensorflow import keras
from tensorflow.keras.layers import *
import numpy as np
import pylab as plt
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras import layers ,models
from tensorflow.keras.layers import BatchNormalization
import os
import re
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import statistics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers ,models
from skimage.transform import rescale
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import LSTM

########### Label Function #############
def read_label(filename):
    x = re.findall("a.._", filename)
    # print(x)
    action_id=x[0][1:3]
    return action_id


############## Load Data ################
label=[]
d=os.listdir('data/')
f=[]
data=[]
len_f=[]

################# video ##################
for file in d:
    frames=[]
    path='data/'+file
    cap=cv2.VideoCapture(path)
    label.append(int(read_label(file)))
    ret,frame=cap.read()
    # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #frames.append(frame)
    while ret:
        frame = cv2.resize(frame,(80,60))
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = rescale(frame, 1, anti_aliasing=False)
        frames.append(frame)
        ret, frame = cap.read()
    # frames=np.asarray(frames)
    data.append(frames[0:299])
    cap.release()


data_new=np.asarray(data)
# print("data train shape before:",data_Train.shape)
data_new=data_new.reshape(data_new.shape[0],data_new.shape[1],14400)
# print("data train shape after:",data_Train.shape)
############ Labels ##############
label_new1 = np.asarray(label)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = label_new1.reshape(len(label_new1), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
label_new= onehot_encoded
########## Split Train & Test ###############
X_train,X_test,y_train,y_test = train_test_split(data_new,label_new,test_size=0.20075757575757574,random_state=0)
print(X_train.shape,"\n",X_test.shape,"\n",y_train.shape)

########### LSTM Model ###################
Lstm_model=models.Sequential()
Lstm_model.add(LSTM(units=128,return_sequences=True,input_shape=(299,14400)))
# Lstm_model.add(BatchNormalization())
# Lstm_model.add(Dropout(0.2))
# Lstm_model.add(LSTM(units=8,return_sequences=True))
# Lstm_model.add(Dropout(0.2))
Lstm_model.add(LSTM(units=128))
Lstm_model.add(Dropout(0.2))
# Lstm_model.add(Dense(300,activation='relu'))
# Lstm_model.add(Dropout(0.2))
# Lstm_model.add(Dense(100,activation='relu'))
# Lstm_model.add(Dropout(0.2))
# Lstm_model.add(Dense(50,activation='relu'))
# Lstm_model.add(Dropout(0.2))
Lstm_model.add(Dense(18,activation='softmax'))
Lstm_model.summary()

################## Compiling a model ########################
optimiz=optimizers.Adam(learning_rate=0.0005)
Lstm_model.compile( optimiz,#
        loss='categorical_crossentropy',
    # loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])


################### Fitting a model ##########################
history=Lstm_model.fit(x = X_train,
          y = y_train,
          epochs = 60,
         batch_size=1,
                       shuffle=True,
         # steps_per_epoch=400,
         # validation_steps=5,
        validation_data = (X_test, y_test))

################# Evaluate Model ################
result=Lstm_model.evaluate(X_test,y_test)
print("accuracy:",result[1],"loss:",result[0])
############### Plot Result #############
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
