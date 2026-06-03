# =============================================================
# Telecom Customer Churn Prediction
# Author : Baldev Rathod  |  GitHub: github.com/yourgithub
# Dataset: 7,043 telecom subscribers (simulated real-world data)
# Goal   : Predict which customers will churn — enabling proactive retention
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                     'axes.titleweight':'bold','figure.dpi':120})
BLUE  = '#2563EB'
RED   = '#EF4444'
GREEN = '#10B981'
GOLD  = '#F59E0B'

OUTPUT = '/home/claude/projects/telecom_churn/outputs'

# ─────────────────────────────────────────────
# 1. LOAD & INSPECT
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("  TELECOM CUSTOMER CHURN PREDICTION — Baldev Rathod")
print("="*60)

df = pd.read_csv('/home/claude/projects/telecom_churn/data/telecom_customers.csv')
print(f"\n[DATA]  Shape     : {df.shape}")
print(f"[DATA]  Churn rate: {df['Churn'].mean():.1%}  ({df['Churn'].sum():,} churned)")
print(f"[DATA]  Nulls     : {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 2. EDA — Business Insight Charts
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Telecom Customer Churn — Exploratory Data Analysis', fontsize=16, y=1.01)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.40)

# 2a — Churn distribution
ax0 = fig.add_subplot(gs[0,0])
sizes  = [df['Churn'].sum(), (df['Churn']==0).sum()]
colors = [RED, BLUE]
ax0.pie(sizes, labels=['Churned','Retained'], autopct='%1.1f%%',
        colors=colors, startangle=90, textprops={'fontsize':11})
ax0.set_title('Overall Churn Split')

# 2b — Churn by Contract type
ax1 = fig.add_subplot(gs[0,1])
ct = df.groupby('Contract')['Churn'].mean().sort_values(ascending=False)*100
bars = ax1.bar(ct.index, ct.values, color=[RED,GOLD,GREEN], width=0.5, edgecolor='white')
ax1.set_title('Churn Rate by Contract Type')
ax1.set_ylabel('Churn Rate (%)')
ax1.set_ylim(0, 60)
for b in bars:
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+1,
             f'{b.get_height():.1f}%', ha='center', fontsize=10, fontweight='bold')
ax1.tick_params(axis='x', rotation=15)

# 2c — Churn by Internet Service
ax2 = fig.add_subplot(gs[0,2])
is_ = df.groupby('InternetService')['Churn'].mean().sort_values(ascending=False)*100
ax2.bar(is_.index, is_.values, color=[RED,GOLD,GREEN], width=0.5, edgecolor='white')
ax2.set_title('Churn Rate by Internet Service')
ax2.set_ylabel('Churn Rate (%)')
ax2.set_ylim(0, 60)
for i,(idx,val) in enumerate(is_.items()):
    ax2.text(i, val+1, f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

# 2d — Tenure distribution by Churn
ax3 = fig.add_subplot(gs[1,0:2])
df[df['Churn']==0]['tenure'].hist(bins=30, alpha=0.7, color=BLUE,  label='Retained', ax=ax3)
df[df['Churn']==1]['tenure'].hist(bins=30, alpha=0.7, color=RED,   label='Churned',  ax=ax3)
ax3.set_title('Tenure Distribution: Churned vs Retained')
ax3.set_xlabel('Tenure (months)')
ax3.set_ylabel('Count')
ax3.legend()

# 2e — Monthly Charges boxplot
ax4 = fig.add_subplot(gs[1,2])
data_box = [df[df['Churn']==0]['MonthlyCharges'], df[df['Churn']==1]['MonthlyCharges']]
bp = ax4.boxplot(data_box, labels=['Retained','Churned'], patch_artist=True,
                 medianprops={'color':'white','linewidth':2})
bp['boxes'][0].set_facecolor(BLUE)
bp['boxes'][1].set_facecolor(RED)
ax4.set_title('Monthly Charges Distribution')
ax4.set_ylabel('Monthly Charges (₹)')

# 2f — Payment method churn
ax5 = fig.add_subplot(gs[2,0:2])
pm = df.groupby('PaymentMethod')['Churn'].mean().sort_values(ascending=False)*100
ax5.barh(pm.index, pm.values, color=[RED,RED,GOLD,GREEN], edgecolor='white')
ax5.set_title('Churn Rate by Payment Method')
ax5.set_xlabel('Churn Rate (%)')
for i,(idx,val) in enumerate(pm.items()):
    ax5.text(val+0.3, i, f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

# 2g — Senior vs Non-senior
ax6 = fig.add_subplot(gs[2,2])
sr = df.groupby('SeniorCitizen')['Churn'].mean()*100
ax6.bar(['Non-Senior','Senior'], sr.values, color=[BLUE, RED], width=0.4, edgecolor='white')
ax6.set_title('Churn Rate: Senior vs Non-Senior')
ax6.set_ylabel('Churn Rate (%)')
for i,v in enumerate(sr.values):
    ax6.text(i, v+0.5, f'{v:.1f}%', ha='center', fontweight='bold')

plt.savefig(f'{OUTPUT}/01_eda_dashboard.png', bbox_inches='tight')
plt.close()
print(f"[EDA]  Dashboard saved.")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
df2 = df.copy()

# encode binary/ordinal
le = LabelEncoder()
binary_cols = ['gender','Partner','Dependents','PaperlessBilling']
for col in binary_cols:
    df2[col] = le.fit_transform(df2[col])

# map service cols
service_map = {'No':0,'No internet service':0,'No phone service':0,'Yes':1}
for col in ['MultipleLines','OnlineSecurity','TechSupport']:
    df2[col] = df2[col].map(service_map)

# ordinal contract
contract_map = {'Month-to-month':0,'One year':1,'Two year':2}
df2['Contract'] = df2['Contract'].map(contract_map)

# internet service dummies
df2 = pd.get_dummies(df2, columns=['InternetService','PaymentMethod'], drop_first=False)
df2.drop(columns=['customerID'], inplace=True)

# engineered features
df2['ChargesPerMonth']   = df2['TotalCharges'] / (df2['tenure'] + 1)
df2['IsNewCustomer']     = (df2['tenure'] <= 3).astype(int)
df2['IsLongTermCustomer']= (df2['tenure'] >= 36).astype(int)
df2['HighValueAtRisk']   = ((df2['MonthlyCharges'] > 80) & (df2['Contract'] == 0)).astype(int)

print(f"[FEAT] Features after engineering: {df2.shape[1]-1}")

X = df2.drop('Churn', axis=1)
y = df2['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"[SPLIT] Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ─────────────────────────────────────────────
# 4. MODEL TRAINING & COMPARISON
# ─────────────────────────────────────────────
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ]),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight='balanced',
        min_samples_leaf=5, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=4,
        subsample=0.8, random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n[MODEL COMPARISON]")
print(f"{'Model':<25} {'CV-AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
print("-"*65)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = (model.predict_proba(X_test)[:,1]
              if hasattr(model,'predict_proba') else None)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
    f1  = f1_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cv_auc = cross_val_score(model, X_train, y_train,
                             cv=cv, scoring='roc_auc').mean()
    results[name] = dict(model=model, auc=auc, f1=f1,
                         precision=pre, recall=rec, cv_auc=cv_auc,
                         y_pred=y_pred, y_prob=y_prob)
    print(f"{name:<25} {cv_auc:>8.4f} {f1:>8.4f} {pre:>10.4f} {rec:>8.4f}")

best_name = max(results, key=lambda k: results[k]['f1'])
best      = results[best_name]
print(f"\n[WINNER] Best model: {best_name}  (F1={best['f1']:.4f}, AUC={best['auc']:.4f})")

# ─────────────────────────────────────────────
# 5. MODEL EVALUATION CHARTS
# ─────────────────────────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle(f'Model Evaluation — {best_name}', fontsize=14)

# Confusion matrix
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Retained','Churned'],
            yticklabels=['Retained','Churned'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# ROC curves
axes[1].plot([0,1],[0,1],'k--',alpha=0.4, label='Random (AUC=0.50)')
colors_ = [BLUE, RED, GREEN]
for i,(name,res) in enumerate(results.items()):
    if res['y_prob'] is not None:
        fpr,tpr,_ = roc_curve(y_test, res['y_prob'])
        axes[1].plot(fpr, tpr, color=colors_[i],
                     label=f"{name} (AUC={res['auc']:.3f})", linewidth=2)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC-AUC Comparison')
axes[1].legend(fontsize=9)

# Feature importance (best model if RF/GB)
best_model = best['model']
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=X.columns)
    fi_top = fi.nlargest(12)
    axes[2].barh(fi_top.index, fi_top.values, color=BLUE, edgecolor='white')
    axes[2].set_title(f'Top 12 Feature Importances\n({best_name})')
    axes[2].set_xlabel('Importance')
elif hasattr(best_model.named_steps.get('clf', best_model), 'coef_'):
    clf = best_model.named_steps['clf']
    coef = pd.Series(abs(clf.coef_[0]), index=X.columns).nlargest(12)
    axes[2].barh(coef.index, coef.values, color=BLUE, edgecolor='white')
    axes[2].set_title('Top 12 Feature Coefficients\n(Logistic Regression)')

plt.tight_layout()
plt.savefig(f'{OUTPUT}/02_model_evaluation.png', bbox_inches='tight')
plt.close()
print(f"[EVAL]  Model evaluation chart saved.")

# ─────────────────────────────────────────────
# 6. BUSINESS IMPACT SUMMARY
# ─────────────────────────────────────────────
total_customers = len(y_test)
actual_churners = y_test.sum()
predicted_churners = best['y_pred'].sum()
true_positives = cm[1][1]

avg_monthly = df['MonthlyCharges'].mean()
revenue_saved = true_positives * avg_monthly * 6  # 6 months retention value

print("\n" + "="*60)
print("  BUSINESS IMPACT SUMMARY")
print("="*60)
print(f"  Test set customers     : {total_customers:,}")
print(f"  Actual churners        : {actual_churners:,}  ({actual_churners/total_customers:.1%})")
print(f"  Correctly identified   : {true_positives:,}  churners caught by model")
print(f"  Model F1-Score         : {best['f1']:.1%}")
print(f"  Model ROC-AUC          : {best['auc']:.1%}")
print(f"  Est. revenue protected : ₹{revenue_saved:,.0f}  (6-month retention value)")
print("="*60)

# ─────────────────────────────────────────────
# 7. SAVE PREDICTIONS
# ─────────────────────────────────────────────
test_df = df.iloc[X_test.index].copy()
test_df['Predicted_Churn'] = best['y_pred']
test_df['Churn_Probability'] = (best['y_prob'] * 100).round(1)
test_df['Risk_Segment'] = pd.cut(best['y_prob'],
                                  bins=[0,0.3,0.6,1.0],
                                  labels=['Low Risk','Medium Risk','High Risk'])
test_df.to_csv(f'{OUTPUT}/churn_predictions.csv', index=False)
print(f"\n[OUTPUT] Predictions saved: {OUTPUT}/churn_predictions.csv")
print("[DONE]  All outputs saved to /outputs/ folder.\n")
