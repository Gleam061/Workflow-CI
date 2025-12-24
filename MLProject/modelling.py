import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import argparse
import os

# --- Ambil argumen dari MLflow Project ---
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="heart_preprocessed.csv")
args = parser.parse_args()

print(f"Loading dataset from: {args.data_path}")

# --- Load dataset ---
data_path = os.path.join(os.path.dirname(__file__), args.data_path)
df = pd.read_csv(data_path)

X = df.drop("target", axis=1)
y = df["target"]

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Evaluate ---
score = model.score(X_test, y_test)
print(f"✅ Model Accuracy: {score}")

# --- Log ke MLflow ---
with mlflow.start_run():
    mlflow.log_param("data_path", args.data_path)
    mlflow.log_metric("accuracy", score)
    mlflow.sklearn.log_model(model, "model")

print("Training complete and model logged to MLflow.")