import pandas as pd
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# -------- Load Resumes -------- #
def load_resumes(processed_dir="data/processed"):
    resumes = []
    files = list(Path(processed_dir).glob("*.json"))
    for f in files:
        obj = json.load(open(f, encoding="utf8"))
        resumes.append({
            "filename": obj["filename"],
            "clean_text": obj.get("clean_text", obj["text"])
        })
    return pd.DataFrame(resumes)

# -------- Retraining Pipeline -------- #
def retrain(processed_dir="data/processed", feedback_file="data/feedback.csv", model_path="models/feedback_model.pkl"):
    # Load feedback
    if not Path(feedback_file).exists():
        print("No feedback file found. Add recruiter feedback first.")
        return
    
    feedback = pd.read_csv(feedback_file)
    resumes = load_resumes(processed_dir)

    # Merge resumes with feedback
    data = resumes.merge(feedback, on="filename", how="inner")
    if data.empty:
        print("No matching resumes found for feedback entries.")
        return

    # Map labels
    data["label"] = data["decision"].map({"selected": 1, "rejected": 0})

    # Build pipeline: TF-IDF + Logistic Regression
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=1000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Train model
    model.fit(data["clean_text"], data["label"])

    # Save model
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model retrained and saved to {model_path}")

if __name__ == "__main__":
    retrain()
