import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

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

# Split data (same split for both datasets)
X1_train, X1_test, X2_train, X2_test, y_train_botanical, y_test_botanical, y_train_geographical, y_test_geographical = train_test_split(
    X1_scaled, X2_scaled, y_botanical, y_geographical, test_size=0.2, random_state=42, stratify=y_botanical
)

# Train separate classifiers
rf1_botanical = RandomForestClassifier(n_estimators=100, random_state=42)
rf1_botanical.fit(X1_train, y_train_botanical)

rf2_botanical = RandomForestClassifier(n_estimators=100, random_state=42)
rf2_botanical.fit(X2_train, y_train_botanical)

rf1_geographical = RandomForestClassifier(n_estimators=100, random_state=42)
rf1_geographical.fit(X1_train, y_train_geographical)

rf2_geographical = RandomForestClassifier(n_estimators=100, random_state=42)
rf2_geographical.fit(X2_train, y_train_geographical)

# Get prediction probabilities from both models
prob1_botanical = rf1_botanical.predict_proba(X1_test)
prob2_botanical = rf2_botanical.predict_proba(X2_test)

prob1_geographical = rf1_geographical.predict_proba(X1_test)
prob2_geographical = rf2_geographical.predict_proba(X2_test)

# Decision-level fusion: Average probabilities
final_prob_botanical = (prob1_botanical + prob2_botanical) / 2
final_prob_geographical = (prob1_geographical + prob2_geographical) / 2

# Convert probabilities to final predictions
final_pred_botanical = np.argmax(final_prob_botanical, axis=1)
final_pred_geographical = np.argmax(final_prob_geographical, axis=1)

# Get class labels
botanical_labels = rf1_botanical.classes_
geographical_labels = rf1_geographical.classes_

# Map indices to class labels
final_pred_botanical = [botanical_labels[i] for i in final_pred_botanical]
final_pred_geographical = [geographical_labels[i] for i in final_pred_geographical]

# Print classification reports
print("Botanical classification report (Decision-Level Fusion):")
print(classification_report(y_test_botanical, final_pred_botanical))

print("Geographical classification report (Decision-Level Fusion):")
print(classification_report(y_test_geographical, final_pred_geographical))
