
# coding: utf-8

# In[1]:


#Load packages
#Data handling
import requests
import numpy as np
import pandas as pd
import string as st
import os
import csv
import re
import random
from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

import seaborn as sns
import matplotlib.pyplot as plt 


# In[2]:


#Set path

path1 = 'DataFinal_clean.csv'

X = pd.read_csv(path1,encoding= "ISO-8859-1", low_memory=False)

# #### Feature Selection

# In[101]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from __future__ import division


# In[102]:


X.head()


# In[103]:


target = X['gestage']
target = pd.to_numeric(target)


# In[104]:


newset = X.drop(['gestage'], axis=1)
X_new = SelectKBest(chi2, k=30).fit_transform(newset, target)


# In[105]:


X_new = pd.DataFrame(X_new)


# In[ ]:

lr = LinearRegression()
rfe = RFE(lr, 1)
rfe = rfe.fit(newset, target) 
print(rfe.support_)
print(rfe.ranking_)


# In[ ]:


columnlist = list(newset.columns)


# In[ ]:


ranks = list(rfe.ranking_)


# In[ ]:


columnlist = pd.DataFrame(columnlist)
columnlist["ranks"] = ranks
topfeatures = columnlist.sort_values("ranks")
topfeatures = topfeatures.rename(index=str, columns={0: "features"})
topfeatures = topfeatures.reset_index()
topfeatures.head()


# In[ ]:


top30 = topfeatures.features[0:30]
top30


# In[ ]:


top20 = topfeatures.features[0:20]
top10 = topfeatures.features[0:10]
top10

