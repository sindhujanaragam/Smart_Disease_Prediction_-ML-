# ============================================================
# Smart Disease Prediction System
# Stacking Ensemble Model
# Existing Voting Ensemble is NOT modified
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ============================================================
# 1. Load Dataset
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv("heart_disease_dataset.csv")

X = df.drop("target", axis=1)
y = df["target"]


# ============================================================
# 2. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. Define Base Models
# ============================================================

svm = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        probability=True,
        random_state=42
    ))
])


knn = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(
        n_neighbors=5
    ))
])


rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


gb = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)


# ============================================================
# 4. Base Estimators
# ============================================================

base_models = [
    ("svm", svm),
    ("knn", knn),
    ("random_forest", rf),
    ("gradient_boosting", gb)
]


# ============================================================
# 5. Create Stacking Ensemble
# ============================================================

print("\nCreating Stacking Ensemble...")

stacking_model = StackingClassifier(
    estimators=base_models,

    final_estimator=LogisticRegression(
        max_iter=1000
    ),

    stack_method="predict_proba",

    cv=5,

    passthrough=False
)


# ============================================================
# 6. Train Stacking Model
# ============================================================

print("Training Stacking Ensemble...")

stacking_model.fit(
    X_train,
    y_train
)


# ============================================================
# 7. Predictions
# ============================================================

y_pred = stacking_model.predict(X_test)

y_probability = stacking_model.predict_proba(X_test)[:, 1]


# ============================================================
# 8. Evaluation
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

cm = confusion_matrix(
    y_test,
    y_pred
)

report = classification_report(
    y_test,
    y_pred
)


# ============================================================
# 9. Display Results
# ============================================================

print("\n")
print("=" * 60)
print("        STACKING ENSEMBLE RESULTS")
print("=" * 60)

print(
    f"\nStacking Ensemble Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Stacking Ensemble ROC-AUC: "
    f"{roc_auc:.3f}"
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)


# ============================================================
# 10. Save Stacking Model
# ============================================================

joblib.dump(
    stacking_model,
    "stacking_disease_model.pkl"
)

print("=" * 60)
print("Stacking model saved successfully!")
print("File: stacking_disease_model.pkl")
print("=" * 60)