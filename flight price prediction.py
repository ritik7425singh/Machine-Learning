#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# In[75]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[76]:


flight=pd.read_excel(r"Data_Train.xlsx")


# In[77]:


sns.set_style('whitegrid')


# In[78]:


sns.pairplot(flight)


# In[79]:


sns.distplot(flight['Price'])


# In[80]:


date=flight['Date_of_Journey']
duration=flight['Duration']
result=flight['Price']
plt.bar(x=date, height=result)


# In[81]:


flight.head()


# In[82]:


flight.tail()


# In[83]:


flight.info()


# In[84]:


flight.isnull()


# In[85]:


flight.isnull().sum()


# In[86]:


flight.dropna(inplace=True)
flight.isnull().sum()


# In[87]:


sns.heatmap(flight.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[88]:


flight['Duration'].value_counts()


# In[89]:


flight['Journey_Day']=pd.to_datetime(flight.Date_of_Journey,format='%d/%m/%Y').dt.day 


# In[90]:


flight['Journey_Month']=pd.to_datetime(flight.Date_of_Journey,format='%d/%m/%Y').dt.month


# In[91]:


flight.head()


# In[92]:


flight.drop(['Date_of_Journey'],axis=1,inplace= True)


# In[93]:


flight['Dep_Hour']=pd.to_datetime(flight['Dep_Time']).dt.hour


# In[94]:


flight['Dep_Min']=pd.to_datetime(flight['Dep_Time']).dt.minute


# In[95]:


flight.drop(['Dep_Time'],axis=1,inplace= True)


# In[96]:


flight.head()


# In[97]:


flight['Arrival_hour']=pd.to_datetime(flight['Arrival_Time']).dt.hour
flight['Arrival_min']=pd.to_datetime(flight['Arrival_Time']).dt.minute
flight.drop(['Arrival_Time'],axis=1,inplace=True)


# In[98]:


flight.head()


# In[99]:


duration=list(flight['Duration'])
for i in range(len(duration)):
    if len(duration[i].split()) !=2:
        if 'h' in duration[i]:
            duration[i]=duration[i].strip()+ ' 0m'
        else:
            duration[i]='0h '+duration[i]

duration_hours=[]
duration_mins=[]
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep='h')[0]))
    duration_mins.append(int(duration[i].split(sep='m')[0].split()[-1]))


# In[100]:


flight['Duration_hrs']=duration_hours
flight['Duration_min']=duration_mins


# In[101]:


flight.drop(['Duration'],axis=1,inplace=True)


# In[102]:


flight.head()


# In[103]:


flight['Airline'].value_counts()


# In[104]:


Airlines=flight[['Airline']]
Airlines=pd.get_dummies(Airlines,drop_first =True)
Airlines.head()


# In[105]:


Sources=flight[['Source']]
Sources=pd.get_dummies(Sources,drop_first=True)
Sources.head()


# In[106]:


flight['Destination'].value_counts()


# In[107]:


Destinations=flight[['Destination']]
Destinations=pd.get_dummies(Destinations,drop_first=True)
Destinations.head()


# In[108]:


flight.drop(['Route','Additional_Info'],axis=1,inplace=True)


# In[109]:


flight.replace({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4},inplace= True)


# In[110]:


flight.head()


# In[111]:


flight_train=pd.concat([flight,Airlines,Sources,Destinations],axis=1)


# In[112]:


flight_train.head()


# In[113]:


flight_train.drop(['Airline','Source','Destination'],axis=1,inplace=True)


# In[114]:


flight_train.head()


# In[115]:


#TEST DATA ANALYSIS


# In[116]:


flight_test=pd.read_excel(r"Test_set.xlsx")


# In[117]:


flight_test.head()


# In[118]:


flight_test.tail()


# In[119]:


flight_test.replace({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4},inplace= True)


# In[120]:


flight_test.head()


# In[121]:


sns.pairplot(flight_test)


# In[122]:


flight_test.isnull()


# In[123]:


flight_test.isnull().sum()


# In[124]:


sns.heatmap(flight_test.isnull(),yticklabels=False,cbar=False,cmap='magma')


# In[125]:


flight_test['Journey_Day']=pd.to_datetime(flight_test.Date_of_Journey,format='%d/%m/%Y').dt.day 
flight_test['Journey_Month']=pd.to_datetime(flight_test.Date_of_Journey,format='%d/%m/%Y').dt.month 
flight_test.head()


# In[126]:


flight_test['Dept_Hour']=pd.to_datetime(flight_test.Dep_Time).dt.hour
flight_test['Dept_Min']=pd.to_datetime(flight_test.Dep_Time).dt.minute
flight_test.head()


# In[127]:


duration=list(flight_test['Duration'])
for i in range(len(duration)):
    if len(duration[i].split()) !=2:
        if 'h' in duration[i]:
            duration[i]=duration[i].strip()+ ' 0m'
        else:
            duration[i]='0h '+duration[i]

duration_hours=[]
duration_mins=[]
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep='h')[0]))
    duration_mins.append(int(duration[i].split(sep='m')[0].split()[-1]))


# In[128]:


flight_test['Duration_hours']=duration_hours
flight_test['Duration_mins']=duration_mins


# In[129]:


flight_test.head()


# In[130]:


flight_test['Arrival_hour']=pd.to_datetime(flight_test['Arrival_Time']).dt.hour
flight_test['Arrival_min']=pd.to_datetime(flight_test['Arrival_Time']).dt.minute
flight_test.drop(['Arrival_Time'],axis=1,inplace=True)


# In[131]:


flight_test.head()


# In[132]:


flight_test.drop(['Additional_Info','Duration','Route','Dep_Time'],axis=1,inplace=True)


# In[133]:


flight_test.head()


# In[134]:


flight_test['Airline'].value_counts()


# In[135]:


airline_test=flight_test['Airline']
airline_test=pd.get_dummies(airline_test,drop_first=True)
airline_test.head()


# In[136]:


source_test=flight_test['Source']
source_test=pd.get_dummies(source_test,drop_first=True)
source_test.head()


# In[137]:


destination_test=flight_test['Destination']
destination_test=pd.get_dummies(destination_test,drop_first=True)
destination_test.head()


# In[138]:


flight_test.drop(['Destination','Source','Date_of_Journey','Airline'],axis=1,inplace=True)


# In[139]:


flight_test1=pd.concat([flight_test,airline_test,source_test,destination_test],axis=1)


# In[140]:


flight_test1.head()


# In[141]:


flight_train.columns


# In[142]:


X=flight_train.loc[:,['Total_Stops', 'Journey_Day', 'Journey_Month', 'Dep_Hour',
       'Dep_Min', 'Arrival_hour', 'Arrival_min', 'Duration_hrs',
       'Duration_min', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
X.head()


# In[143]:


Y=flight_train.loc[:,'Price']
Y.head()


# In[144]:


Y.tail()


# In[145]:


plt.figure(figsize=(18,18))
sns.heatmap(flight.corr(),annot=True,cmap='magma')
plt.show


# In[152]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=51)


# In[153]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()
rfr.fit(x_train,y_train)


# In[154]:


pred=rfr.predict(x_test)


# In[155]:


rfr.score(x_train,y_train)


# In[156]:


rfr.score(x_test,y_test)


# In[159]:


sns.distplot(y_test-pred)


# In[162]:


plt.scatter(y_test,pred,alpha=0.8)
plt.xlabel('y_test')
plt.ylabel('pred')


# In[177]:


plt.figure(figsize=(12,8))
imp=pd.Series(rfr.feature_importances_,index=X.columns)
imp.nlargest(20).plot(kind='bar')
plt.show


# In[163]:


from sklearn import metrics


# In[166]:


print('MAE',metrics.mean_absolute_error(y_test,pred))
print('MSE',metrics.mean_squared_error(y_test,pred))
print('RMSE',np.sqrt(metrics.mean_squared_error(y_test,pred)))


# In[167]:


metrics.r2_score(y_test,pred)


# In[213]:


import pickle
file=open('flight-fare-prediction.pkl','wb')
pickle.dump(rfr,file)


# In[215]:


model=open('flight-fare-prediction.pkl','rb')
mod=pickle.load(model)


# In[216]:


prediction_data=mod.predict(x_test)


# In[217]:


metrics.r2_score(y_test,prediction_data)


# In[ ]:





# In[ ]:




