# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:15:21 2020

@author: Gaurav Jariwala
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data.csv')

df_model = df[['Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Type', 'Price','Content Rating',
       'Android Ver', 'Size_Varies', 'Size_in_Mb', 'Size_in_Kb', 'Photography',
       'Video Players & Editors', 'Events', 'Communication', 'Social',
       'Auto & Vehicles', 'Art & Design', 'Productivity', 'Weather', 'Trivia',
       'Action & Adventure', 'House & Home', 'Music & Video', 'Beauty',
       'Strategy', 'Business', 'Adventure', 'Sports', 'Pretend Play',
       'Music & Audio', 'Maps & Navigation', 'Parenting', 'Word', 'Simulation',
       'Casino', 'Casual', 'Music', 'Racing', 'Brain Games',
       'Libraries & Demo', 'Travel & Local', 'Health & Fitness', 'Educational',
       'Lifestyle', 'Board', 'Personalization', 'Dating', 'Food & Drink',
       'Action', 'Comics', 'Education', 'Puzzle', 'Entertainment', 'Shopping',
       'Role Playing', 'Medical', 'News & Magazines', 'Finance', 'Card',
       'Arcade', 'Creativity', 'Books & Reference', 'Tools',
       'Last Updated Year']]

df_dum = pd.get_dummies(df_model, columns=['Category', 'Installs', 'Type', 'Content Rating', 'Android Ver'], drop_first=True)


# train test split
from sklearn.model_selection import train_test_split

X = df_dum.drop('Rating', axis =1)
y = df_dum.Rating.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# X_train= scaler.fit_transform(X_train)

# X_test = scaler.transform(X_test)


import xgboost
classifier=xgboost.XGBRegressor()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test,y_pred)


sns.heatmap(df[['Rating', 'Reviews', 'Price', 'Last Updated Year']].corr(), annot=True)

sns.heatmap(df.corr(), annot=True)

from sklearn.feature_selection import SelectKBest, chi2


from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV


rfecv = RFECV(estimator=classifier, step=1, cv=5, scoring='neg_mean_absolute_error')
rfecv = rfecv.fit(X_train, y_train)

print('Optimal number of features :', rfecv.n_features_)

best_features = X_train.columns[rfecv.support_]

print('Best features :', best_features)

rfecv.grid_scores_

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

X_train_rfecv = rfecv.transform(X_train)
X_test_rfecv = rfecv.transform(X_test)

lr_rfecv_model = classifier.fit(X_train_rfecv, y_train)

lr_rfecv_pred = lr_rfecv_model.predict(X_test_rfecv)

lr_rfecv_mae = mean_absolute_error(y_test,lr_rfecv_pred)


# Our predictions
plt.scatter(y_test,lr_rfecv_pred)

# Perfect predictions
plt.plot(y_test,y_test,'r')


df_model_sel_feat = df[['Rating', 'Reviews', 'Size', 'Installs', 'Type', 
       'Size_Varies', 'Size_in_Mb', 'Size_in_Kb']]


df_dum_sel_feat = pd.get_dummies(df_model_sel_feat, columns=['Installs', 'Type'], drop_first=True)


# train test split
from sklearn.model_selection import train_test_split

X_sel_feat = df_dum_sel_feat.drop('Rating', axis =1)
y_sel_feat = df_dum_sel_feat.Rating.values

X_train_sel_feat, X_test_sel_feat, y_train_sel_feat, y_test_sel_feat = train_test_split(X_sel_feat, y_sel_feat, test_size=0.2, random_state=42)


# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# X_train= scaler.fit_transform(X_train)

# X_test = scaler.transform(X_test)


import xgboost
reg=xgboost.XGBRegressor()
reg.fit(X_train_sel_feat, y_train_sel_feat)

y_pred_sel_feat = reg.predict(X_test_sel_feat)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test_sel_feat,y_pred_sel_feat)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


model = Sequential()

model.add(Dense(24, activation='relu'))
model.add(Dropout(.2))

model.add(Dense(10, activation='relu'))
model.add(Dropout(.2))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')


from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

model.fit(x=X_train_sel_feat, 
          y=y_train_sel_feat, 
          epochs=600,
          validation_data=(X_test_sel_feat, y_test_sel_feat), verbose=1,
          callbacks=[early_stop]
          )


losses = pd.DataFrame(model.history.history)

losses.plot()

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score

predictions = model.predict(X_test_sel_feat)

mean_absolute_error(y_test_sel_feat,predictions)

# Our predictions
plt.scatter(y_test_sel_feat,predictions)

# Perfect predictions
plt.plot(y_test_sel_feat,y_test_sel_feat,'r')