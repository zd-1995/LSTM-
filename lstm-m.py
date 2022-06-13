"""""
lstm-images
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
###### Train data
label1=[]
d1=os.listdir('dataimg/Train/')
f1=[]
data1=[]
len_f1=[]
###### Test data
label2=[]
d2=os.listdir('dataimg/Test/')
f2=[]
data2=[]
len_f2=[]


################ image #################
##### Train
for file in d1:
 #   frames = []
   path = 'dataimg/Train/' + file
   label1.append(int(read_label(file)))
   image=cv2.imread(path)
   image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   image = rescale(image, 1, anti_aliasing=False)
   image=cv2.resize(image,(80,60))
   data1.append(image)
## to array
data_Train=np.asarray(data1)
print("data train shape before:",data_Train.shape)
data_Train=data_Train.reshape(data_Train.shape[0],4800)
print("data train shape after:",data_Train.shape)

###### Test
for file in d2:
 #   frames = []
   path = 'dataimg/Test/' + file
   label2.append(int(read_label(file)))
   image=cv2.imread(path)
   image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   image = rescale(image, 1, anti_aliasing=False)
   image=cv2.resize(image,(80,60))
   data2.append(image)
## to array
data_Test = np.asarray(data2)
print("data test shape before:",data_Test.shape)
data_Test = data_Test.reshape(data_Test.shape[0], 4800)
print("data test shape after:",data_Test.shape)

############ Labels ##############
##### Train
label_new1 = np.asarray(label1)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = label_new1.reshape(len(label_new1), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
label_Train = onehot_encoded
###### Test
label_new2 = np.asarray(label2)
onehot_encoder2 = OneHotEncoder(sparse=False)
integer_encoded2 = label_new2.reshape(len(label_new2), 1)
onehot_encoded2 = onehot_encoder.fit_transform(integer_encoded2)
label_Test = onehot_encoded2
print("label_train:",label_Train.shape,"\n","label_test:",label_Test.shape)
############ Split Train & Test ###############
# X_train,X_test,y_train,y_test = train_test_split(data_new,label_new,test_size=0.20075757575757574,random_state=0,shuffle=False)
# print(X_train.shape,"\n",X_test.shape,"\n",y_train.shape)
time_steps = 150
features = 4800
##reshape input to be [samples, time steps, features]
X_train = data_Train.reshape(-1 ,time_steps,features)
X_test = data_Test.reshape(-1 ,time_steps,features)

print("data train:",data_Train[301:451],"\n","x_train:",X_train[2])
#######
y_train = label_Train.reshape(-1,time_steps,18)
y_test = label_Test.reshape(-1,time_steps,18)
print("label train:",label_Train[150:300],"\n","label:",y_train[1])

print("x_train:",X_train.shape,"\n","x_test:",X_test.shape,"\n","y_train:",y_train.shape,"\n","y_test:",y_test.shape)

############# LSTM Model ###################
Lstm_model=models.Sequential()
Lstm_model.add(LSTM(units=256,return_sequences=True,input_shape=(time_steps,features)))
Lstm_model.add(Dropout(0.2))
# Lstm_model.add(Embedding(input_dim=(300,42000),output_dim=(128)))
# Lstm_model.add(LSTM(units=32,return_sequences=True))
# Lstm_model.add(Dropout(0.2))
# Lstm_model.add(LSTM(units=32,return_sequences=True))
# Lstm_model.add(Dropout(0.2))
# Lstm_model.add(Flatten())
# Lstm_model.add(LSTM(units=32,return_sequences=True))
# Lstm_model.add(Dense(100,activation='relu', input_shape=(150,12)))
# Lstm_model.add(Dropout(0.2))
Lstm_model.add(TimeDistributed(Dense(300,activation='relu')))
Lstm_model.add(Dense(50,activation='relu'))
Lstm_model.add(Dense(18,activation='softmax'))
Lstm_model.summary()

################### Compiling a model ########################
optimiz=optimizers.Adam(learning_rate=0.001)
Lstm_model.compile( optimiz,#
        loss='categorical_crossentropy',
    # loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])


################### Fitting a model ##########################
history=Lstm_model.fit(x = X_train,
          y = y_train,
          epochs = 20,
         batch_size=1,
                       shuffle=False,
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

#
# model.add(LSTM(units = 40, return_sequences = True, input_shape = (64,85)))
# #model.add(Dropout(0.2))
#
# # Adding a second LSTM layer and Dropout layer
# model.add(LSTM(units = 40, return_sequences = True))
# #model.add(Dropout(0.2))
#
# # Adding a third LSTM layer and Dropout layer
# model.add(LSTM(units = 40, return_sequences = True))
# #model.add(Dropout(0.2))
#
# # Adding a fourth LSTM layer and and Dropout layer
# model.add(LSTM(units = 40))
# model.add(Dropout(0.2))
#
# # Adding the output layer
# # For Full connection layer we use dense
# # As the output is 1D so we use unit=1
# model.add(Dense(units = 13))
# model.summary()
#


