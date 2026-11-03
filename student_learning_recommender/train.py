import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import joblib

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("students.csv")

# 1. Basic data checks and preprocessing
print("Shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
df = df.drop_duplicates(subset="student_id")
target = "best_resource_format"
features = [
    "age", "weekly_study_hours", "attendance_pct", "completion_pct",
    "quiz_score", "interest_area", "preferred_format", "topic", "current_level"
]
X, y = df[features], df[target]

numeric = ["age", "weekly_study_hours", "attendance_pct", "completion_pct", "quiz_score"]
categorical = ["interest_area", "preferred_format", "topic", "current_level"]

preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(n_estimators=250, random_state=42,
                                          class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, pred), 3))
print("\nClassification report:\n", classification_report(y_test, pred, zero_division=0))

# 2. EDA visualizations
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.histplot(df["quiz_score"], bins=15, kde=True)
plt.title("Distribution of student quiz scores")
plt.xlabel("Quiz score")
plt.tight_layout()
plt.savefig("outputs/quiz_score_distribution.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="weekly_study_hours", y="quiz_score", hue="current_level")
plt.title("Study hours vs quiz score")
plt.tight_layout()
plt.savefig("outputs/study_hours_vs_score.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="best_resource_format", order=df["best_resource_format"].value_counts().index)
plt.title("Recommended resource format counts")
plt.tight_layout()
plt.savefig("outputs/resource_format_counts.png")
plt.close()

ConfusionMatrixDisplay.from_predictions(y_test, pred, xticks_rotation=45, cmap="Blues")
plt.title("Model confusion matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

joblib.dump(model, "learning_model.joblib")
print("\nSaved model to learning_model.joblib")
print("Saved charts in outputs/")
