import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ── CLEANING ──────────────────────────────────────────────
# Drop marital status
wave6_final.drop(columns=['marital_status'], inplace=True, errors='ignore')

# Fix BMI
wave6_final['bmi'] = wave6_final['bmi'].where(wave6_final['bmi'].between(15, 50), np.nan)
wave6_final['bmi'].fillna(wave6_final['bmi'].median(), inplace=True)

# Fix inactive
wave6_final['inactive'] = wave6_final['inactive'].replace({'Refusal': np.nan, "Don't know": np.nan})
wave6_final['inactive'].fillna(wave6_final['inactive'].mode()[0], inplace=True)
activity_order = {
    'More than once a week': 0,
    'Once a week': 1,
    'One to three times a month': 2,
    'Hardly ever, or never': 3
}
wave6_final['inactive'] = wave6_final['inactive'].map(activity_order)

# ── TARGET ────────────────────────────────────────────────
wave6_final['target'] = pd.cut(wave6_final['eurod'], bins=[-1, 2, 4, 12], labels=[0, 1, 2])
wave6_final.dropna(subset=['target'], inplace=True)

# ── FEATURES ──────────────────────────────────────────────
X = wave6_final.drop(columns=['mergeid', 'eurod', 'target'])
y = wave6_final['target'].astype(int)

# ── SPLIT ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── RANDOM FOREST ─────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
print("── Random Forest ──")
print(classification_report(y_test, rf.predict(X_test)))
print(confusion_matrix(y_test, rf.predict(X_test)))

# ── XGBOOST ───────────────────────────────────────────────
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
print("── XGBoost ──")
print(classification_report(y_test, xgb.predict(X_test)))
print(confusion_matrix(y_test, xgb.predict(X_test)))

# ── FEATURE IMPORTANCE ────────────────────────────────────
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='RF Feature Importance')
plt.tight_layout()
plt.show()
