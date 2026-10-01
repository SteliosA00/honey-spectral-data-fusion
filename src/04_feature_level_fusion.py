#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, f1_score

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path).set_index("Sample")
    return df

# Load datasets
df1 = load_and_preprocess('C:\\Users\\steli\\OneDrive\\Desktop\\DATASET\\honey_uvvis.csv')
df2 = load_and_preprocess('C:\\Users\\steli\\OneDrive\\Desktop\\DATASET\\honey_ftir.csv')

# Align datasets to ensure same samples
df1, df2 = df1.align(df2, join='inner', axis=0)

# Extract target variables
y_botanical = df1.pop('Botanical')
y_geographical = df1.pop('Geographical')
df2.drop(columns=['Botanical', 'Geographical'], inplace=True)

# Standardize features
scaler = StandardScaler()
X1_scaled = scaler.fit_transform(df1)
X2_scaled = scaler.fit_transform(df2)

# Apply PCA
pca1 = PCA(n_components=0.90)
X1_pca = pca1.fit_transform(X1_scaled)

pca2 = PCA(n_components=0.90)
X2_pca = pca2.fit_transform(X2_scaled)

# Feature-level data fusion
X_fused = pd.concat([pd.DataFrame(X1_pca), pd.DataFrame(X2_pca)], axis=1)

# Train-test split for both targets
X_train, X_test, y_train_botanical, y_test_botanical, y_train_geographical, y_test_geographical = train_test_split(
    X_fused, y_botanical, y_geographical, test_size=0.2, random_state=42 , stratify=y_botanical 
)

# Train Random Forest classifiers
rf_botanical = RandomForestClassifier(n_estimators=100, random_state=42)
rf_botanical.fit(X_train, y_train_botanical)

rf_geographical = RandomForestClassifier(n_estimators=100, random_state=42)
rf_geographical.fit(X_train, y_train_geographical)

# Make predictions and print reports
y_pred_botanical = rf_botanical.predict(X_test)
print("Botanical classification report:")
print(classification_report(y_test_botanical, y_pred_botanical))


y_pred_geographical = rf_geographical.predict(X_test)
print("Geographical classification report:")
print(classification_report(y_test_geographical, y_pred_geographical))


# In[ ]:




