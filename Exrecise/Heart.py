#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.svm import SVC
from sklearn import linear_model
from matplotlib import pyplot as plt


# In[2]:


df = pd.read_csv('./Data Sets/heart.csv')


# In[3]:


df.head()
df.Sex.value_counts()


# In[4]:


plt.pie(df['Sex'].value_counts(), labels = ['M', 'F'], autopct = '%0.2f')
plt.show()


# In[5]:


oe = OrdinalEncoder()
df['Sex'] = oe.fit_transform(df[['Sex']])


# ### MALE = 1
# ### FEMALE = 0

# In[6]:


df.head()


# In[7]:


df['ExerciseAngina'] = oe.fit_transform(df[['ExerciseAngina']])
df.tail()
# 0 = N
# 1 = Y


# In[8]:


df['RestingECG'] = oe.fit_transform(df[['RestingECG']])
df['ST_Slope'] = oe.fit_transform(df[['ST_Slope']])
df.tail()
#   RestingECG  
# 0 = LVH
# 1 = NORMAL
# 2 = ST
#   ST_Slope
# 0 = DOWN
# 1 = FLAT
# 2 = UP


# In[9]:


df['ChestPainType'].unique()


# In[10]:


df['ChestPainType'] = oe.fit_transform(df[['ChestPainType']])
df.tail()


# #### ASY = 0
# #### ATA = 1
# #### NAP = 2
# #### TA = 3

# In[11]:


df.features = df.drop('HeartDisease', axis = 1)


# In[12]:


target = df['HeartDisease']


# In[71]:


X_train, X_test, y_train, y_test = train_test_split(df.features, target, test_size=0.2)


# In[70]:


print("X train", len(X_train))
print("X test:", len(X_test))
print("Y train:", len(y_train))
print("Y test:" ,len(y_test))


# In[83]:


lr = linear_model.LogisticRegression()
rf = RandomForestClassifier(n_estimators=100, max_features=90, random_state=80, oob_score=True, n_jobs=-1, max_leaf_nodes=200)
svm = SVC(gamma=1)


# In[69]:


dtc = DecisionTreeClassifier(max_features=200, max_leaf_nodes=2, max_depth=200)


# In[16]:


cross_val_score(lr, X_train, y_train, cv = 10).mean()


# In[86]:


cross_val_score(rf, X_train, y_train, cv = 10).mean()


# In[47]:


cross_val_score(svm, X_train, y_train, cv = 10)


# In[72]:


cross_val_score(dtc, X_train, y_train, cv = 10)


# In[87]:


rf.fit(X_train, y_train)


# In[88]:


y_predicted = rf.predict(X_test)
print(y_predicted)


# In[89]:


accuracy_score(y_test, y_predicted)


# In[90]:


average_precision_score(y_test, y_predicted)


# In[24]:


import pickle


# In[91]:


pickle.dump(rf, open('model.pkl', 'wb'))

