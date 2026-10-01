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


df = pd.read_csv('C:\\Users\\steli\\OneDrive\\Desktop\\DATASET\\honey_ftir.csv')


# In[3]:


data = df.drop(columns=['Sample'])


# In[4]:


X = data.drop(columns=['Geographical', 'Botanical'])
y_geo = data['Geographical']
y_bot = data['Botanical']


# In[5]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[6]:


pca = PCA(n_components=0.95)  
X_pca = pca.fit_transform(X_scaled)


# In[7]:


X_train_geo, X_test_geo, y_train_geo, y_test_geo = train_test_split(X_pca, y_geo, test_size=0.2, random_state=42)
X_train_bot, X_test_bot, y_train_bot, y_test_bot = train_test_split(X_pca, y_bot, test_size=0.2, random_state=42)


# In[8]:


rfc_geo = RandomForestClassifier(random_state=42)
rfc_geo.fit(X_train_geo, y_train_geo)


# In[9]:


y_pred_geo = rfc_geo.predict(X_test_geo)
print("Classification Report for Geographical:")
print(classification_report(y_test_geo, y_pred_geo))


# In[10]:


rfc_bot = RandomForestClassifier(random_state=42)
rfc_bot.fit(X_train_bot, y_train_bot)


# In[11]:


y_pred_bot = rfc_bot.predict(X_test_bot)
print("\nClassification Report for Botanical:")
print(classification_report(y_test_bot, y_pred_bot))


# In[ ]:




