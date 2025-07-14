# Fraud Transaction Detection System

This project builds a machine learning-based fraud detection system using a simulated transaction dataset. It includes an interactive Streamlit dashboard for visual analytics and live fraud prediction.

---

## 🚀 Features

- Detects fraudulent transactions using machine learning
- Simulates real-world fraud scenarios (high amount, terminal compromise, customer compromise)
- Streamlit dashboard for:
  - Fraud vs legitimate visualizations
  - Daily fraud trends
  - Live model training
  - Real-time prediction from user inputs

---

## 📁 Project Structure
```
fraud_detection/
├── data/                      # Directory with daily .pkl transaction files
├── fraud_dashboard.py         # Streamlit app
├── model_training.py          # Script for training and saving the model
├── trained_model.pkl          # Saved Random Forest model
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 📊 Dataset Overview

Each `.pkl` file contains transaction records with:
- `TRANSACTION_ID`
- `TX_DATETIME`
- `CUSTOMER_ID`
- `TERMINAL_ID`
- `TX_AMOUNT`
- `TX_FRAUD` (0 = legitimate, 1 = fraud)

### Fraud Rules Simulated:
1. **High Amount Fraud**: TX_AMOUNT > 220 → fraud
2. **Terminal Compromise**: Random terminals flagged, fraud for 28 days
3. **Customer Compromise**: 1/3 of next 14 days’ txns are fraud (if TX_AMOUNT is high)

---

## ⚙️ How to Run the Project

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/fraud_detection.git
cd fraud_detection
```

### Step 2: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 3: Run the Dashboard
```bash
streamlit run fraud_dashboard.py
```

Open your browser at `http://localhost:8501`

---

## 🧠 Model Info
- Model: Random Forest Classifier
- Trained on features:
  - TX_AMOUNT
  - HIGH_AMOUNT
  - TERMINAL_RECENT_FRAUD
  - CUSTOMER_TX_COUNT
  - CUSTOMER_AVG_AMOUNT
  - SPEND_SPIKE
- Target: `TX_FRAUD`

---

## 📌 Future Enhancements
- Deploy Streamlit dashboard online
- Add CSV upload for batch fraud detection
- Visual model explanations using SHAP or LIME
- Maintain prediction logs for audit

---

## 🛠️ Tech Stack
- **Python** (Pandas, Scikit-Learn, Joblib)
- **Streamlit** (Dashboard)
- **Matplotlib, Seaborn** (Visualization)

---

## 📷 Sample Dashboard Screenshot
*(Include screenshot here in GitHub repo)*

---

## 📄 License
This project is licensed under the MIT License.

---

## 🙌 Acknowledgments
Inspired by real-world fraud detection challenges and simulated for educational and demonstration purposes.
