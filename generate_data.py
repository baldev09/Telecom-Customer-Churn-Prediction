import pandas as pd
import numpy as np
np.random.seed(42)

n = 7043
tenure      = np.random.exponential(scale=30, size=n).clip(1, 72).astype(int)
monthly     = np.round(np.random.normal(65, 30, n).clip(18, 118), 2)
total       = np.round(monthly * tenure * np.random.uniform(0.9, 1.1, n), 2)
contract    = np.random.choice(['Month-to-month','One year','Two year'], n, p=[0.55,0.25,0.20])
internet    = np.random.choice(['DSL','Fiber optic','No'], n, p=[0.34,0.44,0.22])
payment     = np.random.choice(['Electronic check','Mailed check','Bank transfer','Credit card'], n, p=[0.34,0.23,0.22,0.21])
gender      = np.random.choice(['Male','Female'], n)
senior      = np.random.binomial(1, 0.16, n)
partner     = np.random.choice(['Yes','No'], n)
dependents  = np.random.choice(['Yes','No'], n)
tech_support= np.random.choice(['Yes','No','No internet service'], n, p=[0.29,0.49,0.22])
online_sec  = np.random.choice(['Yes','No','No internet service'], n, p=[0.28,0.50,0.22])
mult_lines  = np.random.choice(['Yes','No','No phone service'], n, p=[0.42,0.48,0.10])
paperless   = np.random.choice(['Yes','No'], n, p=[0.59,0.41])

churn_prob = (
    0.40*(contract=='Month-to-month').astype(float) +
    0.15*(internet=='Fiber optic').astype(float)   +
    0.10*(payment=='Electronic check').astype(float)+
    0.08*(senior==1).astype(float)                 -
    0.20*(tenure>24).astype(float)                 -
    0.10*(tech_support=='Yes').astype(float)        +
    np.random.normal(0,0.10,n)
).clip(0.03,0.95)

churn = (np.random.uniform(0,1,n) < churn_prob).astype(int)

df = pd.DataFrame({
    'customerID':       [f'CUST-{str(i).zfill(5)}' for i in range(n)],
    'gender':           gender,     'SeniorCitizen':    senior,
    'Partner':          partner,    'Dependents':       dependents,
    'tenure':           tenure,     'MultipleLines':    mult_lines,
    'InternetService':  internet,   'OnlineSecurity':   online_sec,
    'TechSupport':      tech_support,'Contract':        contract,
    'PaperlessBilling': paperless,  'PaymentMethod':    payment,
    'MonthlyCharges':   monthly,    'TotalCharges':     total,
    'Churn':            churn
})
df.to_csv('/home/claude/projects/telecom_churn/data/telecom_customers.csv', index=False)
print(f"Rows: {len(df)} | Churn rate: {df['Churn'].mean():.1%}")
