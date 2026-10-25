import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ==========================================
# 1. LOAD DIABETES DATASET
# ==========================================

print("Loading Diabetes dataset...")

df = pd.read_csv("diabetes_dataset.csv")

print("\nDataset Shape:", df.shape)


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# ==========================================
# 3. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. BASE MODELS
# ==========================================

svm_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", SVC(
        probability=True,
        random_state=42
    ))
])


knn_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(
        n_neighbors=5
    ))
])


random_forest_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])


gradient_boosting_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", GradientBoostingClassifier(
        random_state=42
    ))
])


# ==========================================
# 5. STACKING ENSEMBLE
# ==========================================

print("\nCreating Stacking Ensemble...")

stacking_model = StackingClassifier(
    estimators=[
        ("svm", svm_model),
        ("knn", knn_model),
        ("rf", random_forest_model),
        ("gb", gradient_boosting_model)
    ],
    final_estimator=LogisticRegression(
        max_iter=1000
    ),
    stack_method="predict_proba",
    cv=5
)


# ==========================================
# 6. TRAIN STACKING MODEL
# ==========================================

print("Training Stacking Ensemble...")

stacking_model.fit(X_train, y_train)


# ==========================================
# 7. PREDICTION
# ==========================================

y_pred = stacking_model.predict(X_test)
y_prob = stacking_model.predict_proba(X_test)[:, 1]


# ==========================================
# 8. EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n==========================================")
print("DIABETES STACKING ENSEMBLE RESULTS")
print("==========================================")

print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("ROC-AUC:", round(roc_auc, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 9. SAVE MODEL
# ==========================================

joblib.dump(
    stacking_model,
    "diabetes_stacking_model.pkl"
)

print("\n==========================================")
print("STACKING MODEL SAVED SUCCESSFULLY")
print("File: diabetes_stacking_model.pkl")
print("==========================================")