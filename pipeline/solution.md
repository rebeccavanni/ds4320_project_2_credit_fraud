##Solution Pipeline
### Project 2: Credit Card Fraud Detection

**Install Dependencies**

```python
!pip install pymongo
```

**Imports**

```python
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pymongo import MongoClient
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score
)
```

**Data Preparation** (Query MongoDB)

```python
#Query the transactions collection and flatten nested fields into a dataframe.
#set Atlas connection string here
MONGO_URI = 'mongodb+srv://rebeccavannivirginia_db_user:rOQIAA8TtPt8s2jz@cluster0.8tqccv3.mongodb.net/?appName=Cluster0'

#connect and pull all documents
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
collection = client["fraud_detection"]["transactions"]

#query all documents and flatten the nested pca_features sub-document
#exclude _id and only pull fields needed for the model
cursor = collection.find({}, {"_id": 0})
records = list(cursor)
client.close()

#flatten nested pca_features dict into top-level columns (V1–V28)
for rec in records:
    rec.update(rec.pop("pca_features", {}))
    rec.pop("cardholder", None)   # drop cardholder sub-doc (not used in model)
    rec.pop("merchant", None)     # drop merchant sub-doc (not used in model)
    rec.pop("transaction_id", None)

df = pd.DataFrame(records)
df.head()
```

**Data Preparation**

Feature / Label Split & SMOTE

**Rationale:** The dataset is severely imbalanced (~0.17% fraud). I applied SMOTE on the training set only to synthetically oversample the minority class so the model learns to detect fraud rather than always predict legitimate. Don't apply SMOTE to the test set.

```python
#features: Time, Amount, V1–V28
FEATURE_COLS = ["time", "amount"] + [f"V{i}" for i in range(1, 29)]
TARGET_COL = "class"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

#train/test split — stratified to preserve fraud ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Fraud in train (before SMOTE): {y_train.sum()} ({y_train.mean():.4%})")

#apply SMOTE to training set only
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```

**Model**

Random Forest Classifier

**Rationale:** Random Forest is the standard baseline for credit card fraud detection (see background reading). It handles class imbalance well, is robust to outliers in `Amount`, and produces feature importances that help explain predictions — important for a model prioritizing cardholder trust.

```python
# Train a Random Forest with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_res, y_train_res)

# Predict on the held-out test set
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
```

**Visualize Results**

**Rationale:** The confusion matrix directly shows the trade-off between false positives and false negatives. False positives are legitimate transactions wrongly declined which is bad for cardholder experience. False negatives are fraud missed which is bad for financial loss. This is the core tension of the problem statement and the most meaningful visualization for stakeholders.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Credit Card Fraud Detection", fontsize=14, fontweight="bold")

# plotting Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Fraud"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix", fontsize=12)

#Top 15 Feature Importances
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
top15 = importances.sort_values(ascending=True).tail(15)
top15.plot(kind="barh", ax=axes[1], color="steelblue", edgecolor="white")
axes[1].set_title("Top 15 Feature Importances", fontsize=12)
axes[1].set_xlabel("Importance")
axes[1].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))

plt.tight_layout()
plt.savefig("fraud_results.png", dpi=150, bbox_inches="tight")
plt.show()
```
