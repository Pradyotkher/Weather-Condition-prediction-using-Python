#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
from  sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
data_frame = pd.read_csv("C:\\Users\\Pradyot\\Desktop\\testset.csv").head(6000)
X = data_frame.drop(['datetime_utc', ' _conds',' _wdire',' _fog',' _hail',' _heatindexm',' _precipm',' _rain',' _snow',' _thunder',' _tornado',' _vism',' _wspdm',' _windchillm',' _wgustm'], axis = 1)
Y = data_frame[' _conds']
X[' _dewptm'] = X[' _dewptm'].fillna(0)
X[' _hum'] = X[' _hum'].fillna(0)
X[' _pressurem'] = X[' _pressurem'].fillna(0)
X[' _tempm'] = X[' _tempm'].fillna(0)
X[' _wdird'] = X[' _wdird'].fillna(0)
X.info()
print(X)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
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




