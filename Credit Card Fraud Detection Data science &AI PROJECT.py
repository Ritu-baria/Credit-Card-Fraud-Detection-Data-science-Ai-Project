# news name detection.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
#data loadset
df=pd.read_csv("creditcard.csv")
print(df.head())

#EDA
print("Fraud vs Non-Fraud counts:\n", df['Class'].value_counts())
sns.countplot(x='Class', data=df)
plt.title("Fraud Class Distribution")
plt.show()

#process the data
# Scale 'Amount' feature
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Drop 'Time' feature (optional)
df = df.drop(columns=['Time'])

# Define features and target
X = df.drop('Class', axis=1)
y = df['Class']

#Handle Imbalanced Data with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print("Balanced class distribution:\n", pd.Series(y_res).value_counts())

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

#Build & Train XGBoost Classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
#Evaluate the Model
y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))


