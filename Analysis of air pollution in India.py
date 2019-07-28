#!/usr/bin/env python
# coding: utf-8

# # AIR POLLUTION

# In[1]:



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import squarify
import missingno as msno


# In[2]:


plt.style.use('fivethirtyeight')


# In[3]:


air_data = pd.read_csv('data.csv',encoding='latin1')


# ## Understanding our Data

# In[4]:


air_data.head()


# In[5]:


air_data.describe()


# In[6]:


air_data.info()


# In[7]:


#Checking the missing values of all the columns, using missingno
msno.bar(air_data,color=sns.color_palette('winter'))


# One particular column 'pm2_5' has a verylow count,lets see how much of data is missing

# In[ ]:


pmpercent = air_data.pm2_5.isnull().sum()/len(air_data.sampling_date)*100
pmpercent = "{0:.3}".format(pmpercent)
print('Total percentage of missing values of PM2_5 is ',pmpercent,'%')


# Since this feature has 97% of missing values, lets drop this column.

# In[8]:


air_data.drop('pm2_5',axis = 1,inplace=True)


# In[9]:


#Filling the missing values of spm,rspm,so2 and no2
air_data.spm = air_data['spm'].fillna(air_data['spm'].mean())
air_data.rspm = air_data['rspm'].fillna(air_data['rspm'].mean())
air_data.so2 = air_data['so2'].fillna(air_data['so2'].mean())
air_data.no2 = air_data['no2'].fillna(air_data['no2'].mean())


# In[10]:


msno.bar(air_data,color=sns.color_palette('winter'))


# ### Lets see the top 20 states with the highest mean value Sulpher di Oxide concentration

# In[11]:



plt.figure(figsize = (14,14))
so2_level  = air_data.groupby(['state']).mean()['so2'].sort_values(ascending = False).to_frame()
sns.barplot(x = 'so2', y = so2_level.index,data = so2_level,palette='inferno')
plt.title('So2 concentration at the highest level statewise')
plt.xlabel('SO2 level',fontsize = 15)
plt.ylabel('State',fontsize = 15,rotation = 'horizontal')


# So2 Levels are higher in the industrial areas.

# ### Now the top 20 states with the highest mean value of Nitrogen Di-oxide State wise

# In[12]:


plt.figure(figsize = (14,14))
so2_level  = air_data.groupby(['state']).mean()['no2'].sort_values(ascending = False).to_frame()
sns.barplot(x = 'no2', y = so2_level.index,data = so2_level,palette='inferno')
plt.title('No2 concentration at the highest level statewise')
plt.xlabel('NO2 level',fontsize = 15)
plt.ylabel('State',fontsize = 15,rotation = 'horizontal')


# In[13]:


#Lets see the highest value of So2 in chronological order, here my aggregate function is max . 

plt.figure(figsize = (14,14))
so2bycity = air_data.groupby(['location']).max()['so2'].sort_values(ascending=False).to_frame()[:20]
sns.pointplot(x  = so2bycity['so2'], y = so2bycity.index,data = so2bycity)
plt.title('So2 Pollutant City wise')
plt.xlabel('So2 Levels',fontsize=14)
plt.ylabel('State',fontsize = 14,rotation = 'horizontal')


# In[14]:


plt.figure(figsize = (14,14))
no2byloc = air_data.groupby(['location']).max()['no2'].sort_values(ascending=False).to_frame()[:20]
sns.pointplot(x  = no2byloc['no2'], y = no2byloc.index,data = no2byloc,color = 'g',markers='x')
plt.title('No2 Pollutant City wise')
plt.xlabel('No2 Levels',fontsize=14)
plt.ylabel('Location',fontsize = 14,rotation = 'horizontal')


# In[15]:


#Lets see the distributions of SO2,NO2,SPM,RSPM and can observe that its a Right skewed curve


# In[16]:


fig = plt.figure(figsize = (12,3))
plt.subplot(1,4,1)
sns.distplot(air_data['so2'],color='r')
#plt.xscale('log')
plt.subplot(1,4,2)
sns.distplot(air_data['no2'],color='b')
#plt.xscale('log')
plt.subplot(1,4,3)
sns.distplot(air_data['spm'],color='g')
#plt.xscale('log')
plt.subplot(1,4,4)
sns.distplot(air_data['rspm'],color='y')
#plt.xscale('log')


# In[17]:


air_data.head()


# In[18]:


air_data['type'].unique()


# In[19]:


#Cleaning the column values 'type'
air_data['type']=air_data['type'].replace(['Sensitive Area','Sensitive'],(['Sensitive Areas','Sensitive Areas']))
air_data['type']=air_data['type'].replace(['Industrial Area','Industrial'],(['Industrial Areas','Industrial Areas']))
air_data['type']=air_data['type'].replace(['Residential','Residential and others'],(['Residential, Rural and other Areas','Residential, Rural and other Areas']))


# In[20]:


plt.figure(figsize = (8,6))
sns.countplot(air_data['type'])


# In[21]:


air_data['type'].value_counts()


# In[22]:


fig,ax = plt.subplots(1,2,figsize = (12,2))
ax1,ax2 = ax.flatten()
cnt = air_data.groupby(['type']).max()['no2'].sort_values(ascending = False).to_frame()[:50]
sns.barplot(x = cnt['no2'],y = cnt.index,ax = ax1 ,palette= 'winter')
ax1.set_title('No2 levels Comparison with Area Types')

cnt = air_data.groupby(['type']).max()['so2'].sort_values(ascending = False).to_frame()[:50]
sns.barplot(x = cnt['so2'],y = cnt.index,ax = ax2 ,palette=  'inferno')
ax2.set_title('So2 levels Comparison with Area Types')
plt.subplots_adjust(wspace = 0.8)


# In[23]:


air_data.groupby(['type','no2'])['no2'].agg(['max']).sort_values(by = 'max',ascending = False)[:20]


# In[24]:


sns.jointplot(x = 'so2',y= 'no2',data = air_data[air_data['no2']<25],color= 'r',kind = 'scatter',size=8)


# In[25]:


air_data.type.unique()


# In[26]:


no2_data = air_data['no2'].fillna(air_data['no2'].mean())


# In[27]:


no2_data = pd.DataFrame(no2_data,columns=['no2'])


# In[28]:


no2_data.head()


# In[29]:


so2_data = air_data['so2'].fillna(air_data['so2'].mean())


# In[30]:


so2_data = pd.DataFrame(so2_data,columns=['so2'])


# In[31]:


so2_data.head()


# In[32]:


air_data.so2.mean()


# In[33]:


#The mean value of the So2 level is 10.
sns.distplot(so2_data[so2_data['so2']<40],color= 'gray')


# In[34]:


#The mean value of the No2 level is 25.
sns.distplot(no2_data[no2_data['no2']<100],color= 'gray')


# In[37]:


#The mean value of the No2 level is 25 after which the values are constant.
grid = sns.PairGrid(air_data,palette='rainbow')
grid.map(plt.scatter)


# RSPM(Residual Suspended Particulate Matter) as observed in different area types.

# In[40]:


plt.figure(figsize=(8,4))
sns.stripplot(x = 'rspm',y= 'type',data = air_data)


# SPM(Suspended Particulate Matter)  as observed in different Area Types ( using  a box plot) 

# In[41]:


plt.figure(figsize=(8,4))
sns.boxplot(x = 'spm',y= 'type',data = air_data)


# RSPM is the highest in Delhi ,here is the graph showing top 20 states for RSPM . 

# In[42]:




plt.figure(figsize=(12,12))
rspm_data = air_data.groupby(['state']).mean()['rspm'].sort_values(ascending = False).to_frame()
sns.barplot(x = 'rspm' , y = rspm_data.index,data = rspm_data,palette='ocean')


# Top 20 SPM levels observed in states.

# In[43]:


plt.figure(figsize=(12,12))
rspm_data = air_data.groupby(['state']).max()['spm'].sort_values(ascending = False).to_frame()
sns.barplot(x = 'spm' , y = rspm_data.index,data = rspm_data,palette='coolwarm')


# We have the column date , lets extract the year to collate the observations and analyse further

# In[44]:


import datetime as datetime


# In[45]:


air_data['year'] = pd.DatetimeIndex(air_data['date']).year

air_data.head()


# In[46]:


air_data.year = air_data.year.fillna(air_data.year.median())


# In[47]:


air_data.year = air_data.year.astype(int)


# So2 levels as observed till the year 2015

# In[48]:


plt.figure(figsize= (14,6))
sns.barplot(x = air_data['year'],y = air_data['so2'],data = air_data,palette = 'summer')


# RSPM levels over the years

# In[49]:


plt.figure(figsize=(12,6))
sns.pointplot(x = air_data['year'],y=air_data['rspm'],data = air_data,palette='rainbow')


# SPM levels till 2015

# In[50]:


plt.figure(figsize=(12,6))
sns.pointplot(x = air_data['year'],y=air_data['spm'],data = air_data,palette='coolwarm')

