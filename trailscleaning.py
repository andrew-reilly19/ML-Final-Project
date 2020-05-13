#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:11:19 2020

@author: andrew
"""

import pandas as pd

#reading in data
data = pd.read_csv("/Users/andrew/Desktop/NCF_DS/MachineLearning/Project/nationalparktrails.csv")
data = data.drop(['country_name','units','city_name'], axis=1)

#recoding latitude and longitude
def lat(w):
    x = w[4]
    y = x.strip('{}')
    y = y.replace(':',',')
    y = y.split(',')
    return(float(y[1]))

def lng(w):
    x = w[4]
    y = x.strip('{}')
    y = y.replace(':',',')
    y = y.split(',')
    return(float(y[3]))

data['latitude'] = data.apply(lat, axis=1)
data['longitude'] = data.apply(lng, axis=1)

data = data.drop(['_geoloc'], axis=1)

#one-hot-encoding route type
routes = pd.get_dummies(data['route_type'])
data = data.join(routes)
data = data.drop(['route_type'], axis=1)

#one-hot-encoding park
# park = pd.get_dummies(data['area_name'])
# data = data.join(park)
# data = data.drop(['area_name'], axis=1)

# #one-hot-encoding route type
# state = pd.get_dummies(data['state_name'])
# data = data.join(state)
# data = data.drop(['state_name'], axis=1)

#recoding features & activities (essentially one-hot encoding)
fa_set = {'surfing', 'birding', 'scenic-driving', 'dogs-no', 'road-biking', 'city-walk',
          'ice-climbing', 'dogs-leash', 'views', 'snowboarding', 'camping', 'wildlife',
          'fly-fishing', 'backpacking', 'sea-kayaking', 'horseback-riding', 'cross-country-skiing',
          'cave', 'dogs', 'kids', 'hiking', 'hot-springs', 'mountain-biking', 'canoeing',
          'paved', 'strollers', 'rails-trails', 'fishing', 'forest', 'paddle-sports',
          'wild-flowers', 'nature-trips', 'partially-paved', 'bike-touring', 'rock-climbing',
          'off-road-driving', 'lake', 'trail-running', 'skiing', 'ada', 'snowshoeing', 'waterfall',
          'historic-site', 'river', 'beach', 'walking', 'whitewater-kayaking'}

for item in fa_set:
    data[item] = 0

for index, row in data.iterrows():
    features = eval(row[11])
    activities = eval(row[12])
    for item in features:
        data.at[index,item]=1
    for item in activities:
        data.at[index,item]=1

data = data.drop(['features','activities'], axis=1)

#normalize other data for ml
cols_to_norm = ['popularity','length','elevation_gain','avg_rating','num_reviews']

data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

data.to_csv("/Users/andrew/Desktop/NCF_DS/MachineLearning/Project/nationalparktrails_finaldata.csv", na_rep="0")
