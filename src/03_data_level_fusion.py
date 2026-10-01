#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[2]:


df1 = pd.read_csv('C:\\Users\\steli\\OneDrive\\Desktop\\DATASET\\honey_uvvis.csv')
df2 = pd.read_csv('C:\\Users\\steli\\OneDrive\\Desktop\\DATASET\\honey_ftir.csv')


# In[3]:


df = pd.merge(df1, df2, on='Sample', how='inner')


# In[5]:


print(df.columns)


# In[6]:


X = df.drop(columns=['Botanical_x', 'Geographical_x'])

y_botanical = df['Botanical_x']

y_geographical = df['Geographical_x']


# In[8]:


print(X.dtypes)


# In[9]:


X = X.drop(columns=['Sample'])

X_scaled = scaler.fit_transform(X)


# In[10]:


pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)


# In[11]:


X_train, X_test, y_train_botanical, y_test_botanical = train_test_split(X_pca, y_botanical, test_size=0.2, random_state=42)
X_train, X_test, y_train_geographical, y_test_geographical = train_test_split(X_pca, y_geographical, test_size=0.2, random_state=42)


# In[12]:


rf_botanical = RandomForestClassifier(n_estimators=100, random_state=42)
rf_botanical.fit(X_train, y_train_botanical)


# In[13]:


rf_geographical = RandomForestClassifier(n_estimators=100, random_state=42)
rf_geographical.fit(X_train, y_train_geographical)


# In[14]:


y_pred_botanical = rf_botanical.predict(X_test)
y_pred_geographical = rf_geographical.predict(X_test)


# In[15]:


print("Botanical classification report:")
print(classification_report(y_test_botanical, y_pred_botanical))


# In[16]:


print("Geographical classification report:")
print(classification_report(y_test_geographical, y_pred_geographical))


# In[ ]:




