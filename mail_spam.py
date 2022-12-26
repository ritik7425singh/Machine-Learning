#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[22]:


data=pd.read_csv(r'mail_data.csv')


# In[23]:


data.head()


# In[24]:


data.isnull().sum()


# In[41]:


sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[31]:


data.info()
data.shape


# In[28]:


data.dropna(inplace=True)
data.isnull().sum()


# In[30]:


data.info()
data.shape


# In[35]:


data.loc[data['Category']=='spam','Category',]= 0


# In[36]:


data.loc[data['Category']=='ham','Category',]= 1


# In[37]:


data.head()


# In[38]:


X=data['Message']
Y=data['Category']


# In[39]:


print(X)


# In[40]:


print(Y)


# In[42]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=51)


# In[43]:


feature=TfidfVectorizer(min_df=1,stop_words='english',lowercase='True')


# In[46]:


x_train_feature=feature.fit_transform(x_train)
x_test_feature=feature.transform(x_test)
y_train=y_train.astype('int')
y_test=y_test.astype('int')


# In[47]:


print(x_train_feature)


# In[50]:


lr=LogisticRegression()


# In[51]:


lr.fit(x_train_feature,y_train)


# In[53]:


pred=lr.predict(x_test_feature)


# In[56]:


lr.score(x_train_feature,y_train)


# In[57]:


lr.score(x_test_feature,y_test)


# In[60]:


prediction_on_test_data=lr.predict(x_test_feature)
accuracy_on_test_data=accuracy_score(y_test,prediction_on_test_data)
print(accuracy_on_test_data)


# In[67]:


input_mail=["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18"]
input_data=feature.transform(input_mail)
pred=lr.predict(input_data)
print(pred)
if pred==0:
    print("It's a spam")
else:
    print("It's a ham")


# In[ ]:




