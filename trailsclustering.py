#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:42:10 2020

@author: andrew
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns


trailsbase = pd.read_csv("/Users/andrew/Desktop/NCF_DS/MachineLearning/Project/nationalparktrails_finaldata.csv")
random.seed(100)
test_indicies = random.sample(range(3313),331)

#just in case, saving this for later:
#names = trailsbase[["trail_id","name"]].copy()
#rfnames = names.to_numpy()
#rftest_names = rfnames[test_indicies,]
#rftrain_names = np.delete(rfnames, test_indicies, axis=0)

labs = trailsbase["avg_rating"]
trailsdata = trailsbase.drop(["avg_rating","Unnamed: 0","name","trail_id","area_name","state_name"], axis=1)

#transforming the response
klabs = labs.to_numpy()
klabs2 = []
for item in klabs:
    x = round(item*5,2)
    cat = str(x)+" Stars"
    klabs2.append(cat)

klabs2=np.array(klabs2)

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
ktest_data = KNNdata[test_indicies,]
ktest_labs = klabs2[test_indicies,]

ktrain_data = np.delete(KNNdata,test_indicies, axis=0)
ktrain_labs = np.delete(klabs2,test_indicies, axis=0)

#CV
error = []
# Calculating error for K values between 1 and 40
for i in range(1, 80):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(ktrain_data, ktrain_labs)
    pred_i = knn.predict(ktest_data)
    error.append(np.mean(pred_i != ktest_labs))


plt.figure(figsize=(12, 6))
plt.plot(range(1, 80), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


#Fitting the classifier
classifier = KNeighborsClassifier(n_neighbors=40)
classifier.fit(ktrain_data,ktrain_labs)

y_pred = classifier.predict(ktest_data)
from sklearn.metrics import classification_report, confusion_matrix
cm1=confusion_matrix(ktest_labs, y_pred)
print(cm1)
print(classification_report(ktest_labs, y_pred))

# matlpotlib method (confusion matrix visualized as a heatmap)
plt.figure(figsize=(9,9))
plt.imshow(cm1, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 15)
plt.colorbar()
tick_marks = np.arange(9)
plt.xticks(tick_marks, ["0", "1", "2", "2.5", "3", "3.5", "4", "4.5", "5"], rotation=45, size = 9)
plt.yticks(tick_marks, ["0", "1", "2", "2.5", "3", "3.5", "4", "4.5", "5"], size = 9)
plt.tight_layout()
plt.ylabel('Actual Stars', size = 15)
plt.xlabel('Predicted Stars', size = 15)
width, height = cm1.shape
for x in range(width):
  for y in range(height):
      plt.annotate(str(cm1[x][y]), xy=(y, x),
                       horizontalalignment='center',
                       verticalalignment='center')


# seaborn method
# plt.figure(figsize=(9,9))
# sns.heatmap(cm1, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
# plt.ylabel('Actual label');
# plt.xlabel('Predicted label');
# all_sample_title = 'Accuracy Score: {0}'.format(score)
# plt.title(all_sample_title, size = 15);