import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load .pkl files
data_dir = 'data'
pkl_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
df_list = [pd.read_pickle(os.path.join(data_dir, f)) for f in pkl_files[:30]]
df = pd.concat(df_list, ignore_index=True)

# Feature engineering
df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
df['TX_DATE'] = df['TX_DATETIME'].dt.date
df['HIGH_AMOUNT'] = (df['TX_AMOUNT'] > 220).astype(int)
fraud_counts = df[df['TX_FRAUD'] == 1].groupby('TERMINAL_ID').size()
df['TERMINAL_FRAUD_COUNT'] = df['TERMINAL_ID'].map(fraud_counts).fillna(0)

# Model training
features = ['TX_AMOUNT', 'HIGH_AMOUNT', 'TERMINAL_FRAUD_COUNT']
X = df[features]
y = df['TX_FRAUD']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'fraud_model.pkl')

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, model.predict(X_test)))
print("\nClassification Report:\n", classification_report(y_test, model.predict(X_test)))
