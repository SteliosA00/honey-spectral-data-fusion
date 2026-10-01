import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df1 = pd.read_csv('honey_uvvis.csv')
df2 = pd.read_csv('honey_ftir.csv')

df = pd.merge(df1, df2, on='Sample', how='inner')
print(df.columns)

X = df.drop(columns=['Botanical_x', 'Geographical_x'])

y_botanical = df['Botanical_x']
y_geographical = df['Geographical_x']
print(X.dtypes)

X = X.drop(columns=['Sample'])
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train_botanical, y_test_botanical = train_test_split(X_pca, y_botanical, test_size=0.2, random_state=42)
X_train, X_test, y_train_geographical, y_test_geographical = train_test_split(X_pca, y_geographical, test_size=0.2, random_state=42)

rf_botanical = RandomForestClassifier(n_estimators=100, random_state=42)
rf_botanical.fit(X_train, y_train_botanical)

rf_geographical = RandomForestClassifier(n_estimators=100, random_state=42)
rf_geographical.fit(X_train, y_train_geographical)

y_pred_botanical = rf_botanical.predict(X_test)
y_pred_geographical = rf_geographical.predict(X_test)

print("Botanical classification report:")
print(classification_report(y_test_botanical, y_pred_botanical))

print("Geographical classification report:")
print(classification_report(y_test_geographical, y_pred_geographical))
