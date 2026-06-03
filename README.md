# 📉 Telecom Customer Churn Prediction

> **Can we predict which customers will leave — before they do?**
>
> Built by **Baldev Rathod** · Data Analyst & ML Engineer · Pune

---

## 🎯 Problem Statement

Telecom companies lose 15–25% of subscribers every year to churn. Every churned customer means lost recurring revenue. This project builds a machine learning pipeline that identifies at-risk customers *before* they cancel — enabling proactive retention campaigns.

**Business question:** Given a customer's contract type, charges, tenure, and service profile — will they churn in the next billing cycle?

---

## 📊 Dataset

| Property | Value |
|---|---|
| Records | 7,043 telecom subscribers |
| Features | 15 input features |
| Target | Churn (binary: 0 / 1) |
| Churn Rate | 26.0% (class imbalance handled) |
| Source | Simulated from real Telco patterns |

**Key features:** tenure, contract type, monthly charges, internet service, payment method, tech support, senior citizen flag

---

## 🔍 Key EDA Findings

- **Month-to-month contracts** churn at 42% vs 11% for 2-year contracts
- **Fiber optic users** churn at 41% — likely due to higher monthly charges
- **Electronic check payers** have the highest churn rate (45%)
- **Customers with < 3 months tenure** are 3.2× more likely to churn
- **Senior citizens** churn at 41% vs 24% for non-seniors

---

## 🧠 Approach

```
Raw Data → EDA → Feature Engineering → Model Training → Evaluation → Business Recommendations
```

**Feature Engineering:**
- `ChargesPerMonth` = TotalCharges / tenure (usage intensity)
- `IsNewCustomer` = tenure ≤ 3 months
- `IsLongTermCustomer` = tenure ≥ 36 months
- `HighValueAtRisk` = high monthly charge + month-to-month contract

**Models compared:**
| Model | CV-AUC | F1-Score |
|---|---|---|
| Logistic Regression | 0.767 | **0.561** ✅ |
| Random Forest | 0.775 | 0.549 |
| Gradient Boosting | 0.769 | 0.426 |

**Winner: Logistic Regression** — Best F1 score, interpretable for business

---

## 📈 Results

| Metric | Value |
|---|---|
| ROC-AUC | 76.8% |
| F1-Score | 56.1% |
| Recall (churn detection) | 78.2% |
| Churners correctly identified | 287 / 367 |
| Estimated revenue protected | ₹1,11,746 (6-month value) |

---

## 💡 Business Recommendations

1. **Offer contract upgrades** to month-to-month customers in months 1–3 (highest churn window)
2. **Proactive retention for fiber optic users** with high monthly charges > ₹80
3. **Switch electronic check payers** to auto-debit to reduce payment friction
4. **Senior citizen retention program** — dedicated support line + loyalty benefits

---

## 🗂️ Project Structure

```
telecom_churn/
├── data/
│   └── telecom_customers.csv       # 7,043 subscriber records
├── src/
│   ├── generate_data.py            # Dataset generation script
│   └── churn_analysis.py           # Main analysis pipeline
├── outputs/
│   ├── 01_eda_dashboard.png        # 7-panel EDA visualization
│   ├── 02_model_evaluation.png     # Confusion matrix + ROC + feature importance
│   └── churn_predictions.csv       # Predictions with risk segments
└── README.md
```

---

## 🛠️ Tech Stack

`Python 3.11` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn`

---

## 🚀 How to Run

```bash
git clone https://github.com/yourgithub/telecom-churn-prediction
cd telecom-churn-prediction
pip install pandas numpy scikit-learn matplotlib seaborn
python3 src/generate_data.py
python3 src/churn_analysis.py
```

---

## 📬 Contact

**Baldev Rathod** | okbaldev123@gmail.com | [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourgithub)
