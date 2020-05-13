#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:01:35 2020

@author: andrew
"""

import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from livelossplot import PlotLossesKeras
from tensorflow.keras.optimizers import Adam


trailsbase = pd.read_csv("/Users/andrew/Desktop/NCF_DS/MachineLearning/Project/nationalparktrails_finaldata.csv")
random.seed(100)
test_indicies = random.sample(range(3313),331)

labs = trailsbase["avg_rating"]
trailsdata = trailsbase.drop(["avg_rating","Unnamed: 0","name","trail_id","area_name","state_name"], axis=1)

#transforming the response
klabs = labs.to_numpy()
klabs2 = []
for item in klabs:
    x = round(item*5,2)
    cat = str(x)
    klabs2.append(cat)

klabs2=np.array(klabs2)
klabs2 = pd.DataFrame(klabs2)
klabs2.columns=['Stars']
klabs3 = pd.get_dummies(klabs2)
labs2 = klabs3.to_numpy()


#recoding variables:
kdata = trailsdata.to_numpy()
difficulty = []
usage = []
for i in range(len(kdata)):
    diff = kdata[i,3]
    if diff == 1:
        difficulty.append("Easy")
    if diff == 3:
        difficulty.append("Moderate")
    if diff == 5:
        difficulty.append("Hard")
    if diff == 7:
        difficulty.append("VHard")
    if diff not in [1,3,5,7]:
        print("ERROR", i)
    use = kdata[i,4]
    if use == 0:
        usage.append("VLight")
    if use == 1:
        usage.append("Light")
    if use == 2:
        usage.append("Moderate")
    if use == 3:
        usage.append("Heavy")
    if use == 4:
        usage.append("VHeavy")
    if use not in [0,1,2,3,4]:
        print("ERROR", i)

difficulty = pd.DataFrame(difficulty)
difficulty.columns=['difficulty']
difficulty1 = pd.get_dummies(difficulty)
diff = difficulty1.to_numpy()

usage = pd.DataFrame(usage)
usage.columns=['Usage']
usage1 = pd.get_dummies(usage)
use = usage1.to_numpy()

kdata1=np.c_[kdata, diff]
kdata2=np.c_[kdata1, use]

KNNdata = np.delete(kdata2, [3,4],axis=1)


#Train/test split:
test_data = KNNdata[test_indicies,]
test_labs = labs2[test_indicies,]

train_data = np.delete(KNNdata,test_indicies, axis=0)
train_labs = np.delete(labs2,test_indicies, axis=0)

model = keras.Sequential()
model.add(Dense(75, input_dim=65))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(.4))
model.add(Dense(75))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(.4))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#model.fit(train_data, train_labs, validation_split=.1, epochs=100)
# print(" ")
# print("Test Data Evaluation:")


preds = model.predict(test_data)

history = model.fit(train_data, train_labs, epochs=20, validation_split=.2)

plt.clf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Categorical Crossentropy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

print(" ")
print("Test Data Evaluation:")

model.evaluate(test_data, test_labs, verbose=2)

#Confusion matrix
pred = model.predict(test_data)




c_preds = []
for item in pred:
    x = item.tolist()
    m = x.index(max(x))
    c_preds.append(m)


c_labs = []
for item in test_labs:
    x = item.tolist()
    m = x.index(max(x))
    c_labs.append(m)





cm2=confusion_matrix(c_labs, c_preds)
print(cm2)
print(classification_report(c_labs, c_preds))
plt.clf()
plt.imshow(cm2, cmap='Pastel1')
plt.xlabel("Predicted Stars")
plt.ylabel("True Stars")
tick_marks = np.arange(9)
plt.xticks(tick_marks, ["0", "1", "2", "2.5", "3", "3.5", "4", "4.5", "5"])
plt.yticks(tick_marks, ["0", "1", "2", "2.5", "3", "3.5", "4", "4.5", "5"])
plt.title('Confusion matrix ')
plt.colorbar()
width, height = cm2.shape
for x in range(width):
  for y in range(height):
      plt.annotate(str(cm2[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
plt.show()



'''
BREAK
'''





#Method for numeric star ratings: (not the best)

#nnlabs = nnlabs*10

test_data2 = test_data
test_labs2 = klabs[test_indicies,]
train_data2 = train_data
train_labs2 = np.delete(klabs,test_indicies, axis=0)

model2 = keras.Sequential()
model2.add(Dense(75, input_dim=65))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(Dropout(.4))
model2.add(Dense(75))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(Dropout(.4))
model2.add(Dense(1))

#optimizer=Adam(lr=0.001)
model2.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

model2.summary()


history = model2.fit(train_data2, train_labs2, epochs=40, validation_split=.2)

plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('Model error')
plt.ylabel('Mean Squared Error')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

print(" ")
print("Test Data Evaluation:")

model2.evaluate(test_data2, test_labs2, verbose=2)




#Confusion matrix
preds2 = model2.predict(test_data2)

preds2 = np.array(preds2)
preds2 = np.round(preds2,2)
preds3 = preds2*10
preds3 = np.round(preds3,0)
preds3 = preds3/2

c_preds = []
for item in preds3:
    x = item[0]
    cat = str(x+.5)+" Stars"
    c_preds.append(cat)


c_labs = []
for item in test_labs2:
    x = item*5
    cat = str(x)+" Stars"
    c_labs.append(cat)




cm2=confusion_matrix(c_labs, c_preds)
print(cm2)
print(classification_report(c_labs, c_preds))
plt.clf()
plt.imshow(cm2, cmap='Pastel1')
plt.xlabel("Predicted Stars")
plt.ylabel("True Stars")
tick_marks = np.arange(9)
plt.xticks(tick_marks, ["0", "1", "2", "2.5", "3", "3.5", "4", "4.5", "5"])
plt.yticks(tick_marks, ["0", "1", "2", "2.5", "3", "3.5", "4", "4.5", "5"])
plt.title('Confusion matrix ')
plt.colorbar()
width, height = cm2.shape
for x in range(width):
  for y in range(height):
      plt.annotate(str(cm2[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
plt.show()
