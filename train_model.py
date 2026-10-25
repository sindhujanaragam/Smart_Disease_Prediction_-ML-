import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# ------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------

df = pd.read_csv("heart_disease_dataset.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# ------------------------------------------------
# 2. Separate Features and Target
# ------------------------------------------------

X = df.drop("target", axis=1)
y = df["target"]

# ------------------------------------------------
# 3. Train-Test Split
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ------------------------------------------------
# 4. Create Models
# ------------------------------------------------

svm = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(probability=True, random_state=42))
])

knn = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(n_neighbors=5))
])

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

gb = GradientBoostingClassifier(
    random_state=42
)

# ------------------------------------------------
# 5. Voting Ensemble
# ------------------------------------------------

ensemble = VotingClassifier(
    estimators=[
        ("svm", svm),
        ("knn", knn),
        ("rf", rf),
        ("gb", gb)
    ],
    voting="soft"
)

# ------------------------------------------------
# 6. Train Model
# ------------------------------------------------

ensemble.fit(X_train, y_train)

# ------------------------------------------------
# 7. Prediction
# ------------------------------------------------

y_pred = ensemble.predict(X_test)
y_probability = ensemble.predict_proba(X_test)[:, 1]

# ------------------------------------------------
# 8. Accuracy
# ------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed!")
print("-------------------------")
print("Voting Ensemble Accuracy:",
      round(accuracy * 100, 2), "%")

# ------------------------------------------------
# 9. Classification Report
# ------------------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------------------------
# 10. Confusion Matrix
# ------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Voting Ensemble")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ------------------------------------------------
# 11. ROC Curve
# ------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

auc_score = roc_auc_score(
    y_test,
    y_probability
)

print("ROC-AUC Score:",
      round(auc_score, 3))

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label="Voting Ensemble (AUC = {:.3f})".format(auc_score)
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# ------------------------------------------------
# 12. Random Forest Feature Importance
# ------------------------------------------------

rf.fit(X_train, y_train)

importance = rf.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(9, 6))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ------------------------------------------------
# 13. SHAP Explainability
# ------------------------------------------------

print("\nGenerating SHAP Explainability...")

# Use Random Forest for SHAP explanation
explainer = shap.TreeExplainer(rf)

shap_values = explainer.shap_values(X_test)

# Handle SHAP output for binary classification
if isinstance(shap_values, list):
    shap_data = shap_values[1]
else:
    shap_data = shap_values

# SHAP Summary Plot
plt.figure()

shap.summary_plot(
    shap_data,
    X_test,
    show=False
)

plt.title("SHAP Feature Importance")

plt.tight_layout()

plt.savefig(
    "shap_summary.png",
    bbox_inches="tight"
)

plt.show()

print("SHAP graph saved as shap_summary.png")

# ==========================================
# ACCURACY COMPARISON: HEART DISEASE
# ==========================================

# Train each individual model on the same training set
svm.fit(X_train, y_train)
knn.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

models = {
    "SVM": svm,
    "KNN": knn,
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "Voting Classifier": ensemble
}

print("\nHEART DISEASE MODEL ACCURACY")
print("=" * 40)

for name, model in models.items():
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * 100
    print(f"{name}: {accuracy:.2f}%")
# ------------------------------------------------
# 14. Save Model
# ------------------------------------------------

joblib.dump(
    ensemble,
    "disease_model.pkl"
)

print("\nModel saved as disease_model.pkl")

print("\nEvaluation graphs saved:")
print("1. confusion_matrix.png")
print("2. roc_curve.png")
print("3. feature_importance.png")
print("4. shap_summary.png")