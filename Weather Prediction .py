#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
from  sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as mb
from sklearn import tree
data_frame = pd.read_csv("C:\\Users\\Pradyot\\Desktop\\testset.csv").head(120)
X = data_frame.drop(['datetime_utc', ' _conds',' _wdire',' _fog',' _hail',' _heatindexm',' _precipm',' _rain',' _snow',' _thunder',' _tornado',' _vism',' _wspdm',' _windchillm',' _wgustm'], axis = 1)
Y = data_frame[' _conds']
data_frame[' _dewptm'] = data_frame[' _dewptm'].fillna(0)
data_frame[' _hum'] = data_frame[' _hum'].fillna(0)
data_frame[' _pressurem'] = data_frame[' _pressurem'].fillna(0)
data_frame[' _tempm'] = data_frame[' _tempm'].fillna(0)
data_frame[' _wdird'] = data_frame[' _wdird'].fillna(0)
print(X)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
model = DecisionTreeClassifier()
model.fit(X_train,Y_train)
predictions = model.predict(X_test)
print("Predictions are as follows: \n")
print(predictions)
acc = accuracy_score(predictions,Y_test)
print("Accuracy: ")
print(acc)


# In[ ]:





# In[ ]:




