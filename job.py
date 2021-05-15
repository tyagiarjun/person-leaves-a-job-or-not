#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('HR_sep.csv')
df.head()


# In[4]:


df.groupby('left').mean()


# In[5]:


pd.crosstab(df.salary,df.left).plot(kind='bar')


# In[6]:


pd.crosstab(df.Department,df.left).plot(kind='bar')


# In[7]:


from sklearn.linear_model import LogisticRegression


# In[8]:


subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
subdf.head()


# In[9]:


salary_dummies=pd.get_dummies(subdf.salary)


# In[10]:


salary_dummies.head()


# In[14]:


merged=pd.concat([subdf,salary_dummies],axis='columns')


# In[15]:


merged.head()


# In[16]:


final=merged.drop(['salary','high'],axis='columns')
final.head()


# In[20]:


X=final
y=df.left


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)


# In[24]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)


# In[26]:


model.score(X_test,y_test)


# In[27]:


from sklearn.externals import joblib


# In[28]:


joblib.dump(model,'binary_classification')


# In[ ]:




