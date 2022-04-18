#!/usr/bin/env python
# coding: utf-8

# In[71]:


#install libraries and dependencies
import numpy as np
import pandas as pd


# In[72]:


#Load datasets
df = pd.read_csv('/Users/Celia/desktop/Current course/data enginering & mining/datasets/BAC.csv')
df.head(10)


# In[73]:


#install libraries and dependencies
from matplotlib import pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[74]:


#data cleaning, check null or not
df.isnull().sum()


# In[75]:


#Count data shape
df.shape






