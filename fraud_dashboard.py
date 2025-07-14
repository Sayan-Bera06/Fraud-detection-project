import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load & prepare data
@st.cache_data
def load_data():
    data_dir = 'data'
    pkl_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
    df_list = [pd.read_pickle(os.path.join(data_dir, f)) for f in pkl_files[:30]]
    df = pd.concat(df_list, ignore_index=True)

    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
    df['TX_DATE'] = df['TX_DATETIME'].dt.date
    df['HIGH_AMOUNT'] = (df['TX_AMOUNT'] > 220).astype(int)
    fraud_counts = df[df['TX_FRAUD'] == 1].groupby('TERMINAL_ID').size()
    df['TERMINAL_FRAUD_COUNT'] = df['TERMINAL_ID'].map(fraud_counts).fillna(0)

    return df

# Load data
df = load_data()

st.title("💳 Fraud Detection Dashboard")

# Metrics
st.subheader("Dataset Summary")
col1, col2 = st.columns(2)
col1.metric("Total Transactions", len(df))
col2.metric("Fraud Cases", df['TX_FRAUD'].sum())

# Fraud distribution
st.subheader("Fraud vs Legitimate")
fig, ax = plt.subplots()
sns.countplot(x='TX_FRAUD', data=df, ax=ax)
ax.set_xticklabels(['Legit (0)', 'Fraud (1)'])
st.pyplot(fig)

# Trend over time
st.subheader("Fraud Trend Over Time")
fraud_trend = df.groupby('TX_DATE')['TX_FRAUD'].sum()
fig2, ax2 = plt.subplots()
fraud_trend.plot(ax=ax2)
ax2.set_ylabel("Fraud Count")
st.pyplot(fig2)

# Train model live
features = ['TX_AMOUNT', 'HIGH_AMOUNT', 'TERMINAL_FRAUD_COUNT']
X = df[features]
y = df['TX_FRAUD']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
joblib.dump(model, 'fraud_model.pkl')

st.success("✅ Model trained!")

# Live prediction
st.subheader("🧠 Predict a Transaction")

tx_amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
high_amount = int(tx_amount > 220)
terminal_fraud_count = st.number_input("Terminal Fraud Count (past)", min_value=0, value=0)

input_df = pd.DataFrame([[tx_amount, high_amount, terminal_fraud_count]], columns=features)

if st.button("Predict Fraud?"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ This transaction is FRAUDULENT!")
    else:
        st.success("✅ This transaction is LEGITIMATE.")
