#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:37:33 2020

@author: andrew
"""

import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
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

rflabs = labs.to_numpy()
rflabs = rflabs + 1
rfdatabase = trailsdata.to_numpy()

#recoding variables:
kdata = trailsdata.to_numpy()
difficulty = []
usage = []
for i in range(len(rfdatabase)):
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

kdata1=np.c_[rfdatabase, diff]
kdata2=np.c_[kdata1, use]

rfdata = np.delete(kdata2, [3,4],axis=1)

rftest_data = rfdata[test_indicies,]
rftest_labs = rflabs[test_indicies,]

rftrain_data = np.delete(rfdata,test_indicies, axis=0)
rftrain_labs = np.delete(rflabs,test_indicies, axis=0)

#establishing baseline
avgrating = sum(labs)/len(labs)
print(avgrating*5)
#average is about 4.17

#finding the error for guessing 4.17 on each:
avgerr = sum(abs(labs-avgrating))/len(labs)
print(avgerr*5)
#Average prediction is .57, a bit over a half star


# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 100)

# Train the model on training data
rf.fit(rftrain_data, rftrain_labs);

# Use the forest's predict method on the test data
predictions = rf.predict(rftest_data)

# Calculate the absolute errors
errors = abs(predictions - rftest_labs)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'stars')
#Mean error is now .31 stars, which is about half of the error we had before!

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / rftest_labs)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

preds = np.round(predictions,1)

c_preds = []
for item in preds:
    x = round((item-1)*5,1)
    cat = str(x)+" Stars"
    c_preds.append(cat)

c_labs = []
for item in rftest_labs:
    x = round((item-1)*5,1)
    cat = str(x)+" Stars"
    c_labs.append(cat)


cm2=confusion_matrix(c_labs, c_preds)
print(cm2)
print(classification_report(c_labs, c_preds))

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




# '''
# Now, for popularity!
# '''

# labs2 = trailsbase["popularity"]
# trailsdata2 = trailsbase.drop(["popularity","Unnamed: 0","name","trail_id","area_name","state_name"], axis=1)

# rflabs2 = labs2.to_numpy()
# rflabs2 = rflabs2 + 1
# rfdata2 = trailsdata2.to_numpy()

# rftest_data2 = rfdata2[test_indicies,]
# rftest_labs2 = rflabs2[test_indicies,]

# rftrain_data2 = np.delete(rfdata2,test_indicies, axis=0)
# rftrain_labs2 = np.delete(rflabs2,test_indicies, axis=0)

# #establishing baseline
# avgrating2 = sum(labs2)/len(labs2)
# #print(avgrating*5)
# #average is about 4.17

# #finding the error for guessing 4.17 on each:
# avgerr2 = sum(abs(labs2-avgrating2))/len(labs2)
# #print(avgerr*5)
# #Average prediction is .57, a bit over a half star


# # Instantiate model with 1000 decision trees
# rf2 = RandomForestRegressor(n_estimators = 1000, random_state = 100)

# # Train the model on training data
# rf2.fit(rftrain_data2, rftrain_labs2);

# # Use the forest's predict method on the test data
# predictions2 = rf.predict(rftest_data2)

# # Calculate the absolute errors
# errors2 = abs(predictions2 - rftest_labs2)

# # Print out the mean absolute error (mae)
# print('Mean Absolute Error:', round(np.mean(errors2), 2), 'popularity score')
# #Mean error is now .31 stars, which is about half of the error we had before!

# # Calculate mean absolute percentage error (MAPE)
# mape2 = 100 * (errors2 / rftest_labs2)

# # Calculate and display accuracy
# accuracy2 = 100 - np.mean(mape2)
# print('Accuracy:', round(accuracy2, 2), '%.')




'''
XGBOOST
'''

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# rftest_data = rfdata[test_indicies,]
# rftest_labs = rflabs[test_indicies,]

# rftrain_data = np.delete(rfdata,test_indicies, axis=0)
# rftrain_labs = np.delete(rflabs,test_indicies, axis=0)

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.5, learning_rate = 0.1,
                          max_depth = 100, alpha = 50, n_estimators = 50)

xg_reg.fit(rftrain_data,rftrain_labs)

preds = xg_reg.predict(rftest_data)

rmse = np.sqrt(mean_squared_error(rftest_labs, preds))
print("RMSE: %f" % (rmse))


preds = np.round(preds,1)

c_preds = []
for item in preds:
    x = round((item-1)*5,1)
    cat = str(x)+" Stars"
    c_preds.append(cat)

c_labs = []
for item in rftest_labs:
    x = round((item-1)*5,1)
    cat = str(x)+" Stars"
    c_labs.append(cat)


cm2=confusion_matrix(c_labs, c_preds)
print(cm2)
print(classification_report(c_labs, c_preds))

plt.imshow(cm2, cmap='Pastel1')
plt.xlabel("Predicted Stars")
plt.ylabel("True Stars")
tick_marks = np.arange(10)
plt.xticks(tick_marks, ["0", "1","1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5"])
plt.yticks(tick_marks, ["0", "1","1.5","2", "2.5", "3", "3.5", "4", "4.5", "5"])
plt.title('Confusion matrix ')
plt.colorbar()
width, height = cm2.shape
for x in range(width):
  for y in range(height):
      plt.annotate(str(cm2[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
plt.show()