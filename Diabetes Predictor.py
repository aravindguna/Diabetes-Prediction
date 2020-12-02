#!/usr/bin/env python
# coding: utf-8

# Importing essential libraries

# In[1]:


import numpy as np
import pandas as pd
import pickle
import sklearn.metrics 
import matplotlib.pyplot as plt
import seaborn as sns


# Loading the dataset

# In[2]:


df = pd.read_csv('kaggle_diabetes.csv')


# In[3]:


df.head(5)


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.isna().sum()


# In[7]:


x = df.drop(columns='Outcome')
y = df['Outcome']


# In[8]:


sns.pairplot(df,hue="Outcome")


# In[9]:


x = df.drop(columns='Outcome')
y = df['Outcome']
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(x,y)
model.feature_importances_


# In[10]:


graph = pd.Series(model.feature_importances_,index = x.columns)
graph.sort_values(inplace = True)
graph.plot(kind = "barh",figsize=(8,5))


# In[11]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


# # Creating Random Forest Model

# # n_estimators

# In[12]:


result=[]
estimators=[10,20,50,100,500,1000,2000,3000]
for variables in estimators:
    model = RandomForestClassifier(n_estimators=variables,oob_score=True,random_state=100)
    model.fit(x_train,y_train)
    print("variables",variables)
    oob = model.oob_score_
    print("OOB",model.oob_score_)
    result.append(oob)
graph = pd.Series(result, estimators)
graph.plot(figsize=(15,5))


# # max_features

# In[13]:


result=[]
features=["auto", None, "sqrt", "log2", 0.9, 0.2,1]
for variables in features:
    model = RandomForestClassifier(n_estimators=100,max_features=variables,random_state=100,oob_score=True)
    model.fit(x_train,y_train)
    print("features_value :",variables)
    oob=model.oob_score_
    print("oob :",oob)
    result.append(oob)
    
pd.Series(result,features).plot(kind='barh', xlim=(.90, 1))


# # min_samples_leaf

# In[14]:


result=[]
features=[1,2,3,4,5,6,7,8,9,10]
for variables in features:
    model = RandomForestClassifier(n_estimators=100,max_features="sqrt",random_state=100,oob_score=True,min_samples_leaf=variables)
    model.fit(x_train,y_train)
    print("features_value :",variables)
    oob=model.oob_score_
    print("oob :",oob)
    result.append(oob)
    
pd.Series(result,features).plot()


# # Final Model

# In[15]:


model = RandomForestClassifier(n_estimators=100, 
                              oob_score=True, 
                              n_jobs=-1, 
                              random_state=100, 
                              max_features="sqrt", 
                              min_samples_leaf=1)
model.fit(x, y)
OOB = model.oob_score_
print('OOB: ', OOB)


# In[16]:


y_pred = model.predict(x_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))


# # Creating a pickle file for the classifier

# In[17]:


filename = 'diabetes-predictor-model.pkl'
pickle.dump(model, open(filename, 'wb'))

