import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib
from pathlib import Path

def load_resumes(processed_dir="data/processed"):
    resumes = []
    files = list(Path(processed_dir).glob("*.json"))
    for f in files:
        obj = json.load(open(f, encoding="utf8"))
        resumes.append({
            "filename": obj["filename"],
            "clean_text": obj.get("clean_text", obj["text"])
        })
    return resumes

def rank_resumes(job_description, resumes, top_n=10):
    docs = [job_description] + [r["clean_text"] for r in resumes]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]
    scores = cosine_similarity(jd_vector, resume_vectors)[0]

    # attach similarity score
    for i, r in enumerate(resumes):
        r["similarity_score"] = scores[i]

    # If feedback model exists, predict probability
    model_path = Path("models/feedback_model.pkl")
    if model_path.exists():
        model = joblib.load(model_path)
        preds = model.predict_proba([r["clean_text"] for r in resumes])[:, 1]
        for i, r in enumerate(resumes):
            r["feedback_score"] = preds[i]
        # Combine scores (50% similarity + 50% feedback)
        for r in resumes:
            r["final_score"] = 0.5 * r["similarity_score"] + 0.5 * r["feedback_score"]
    else:
        for r in resumes:
            r["final_score"] = r["similarity_score"]

    ranked = sorted(resumes, key=lambda x: x["final_score"], reverse=True)
    return ranked[:top_n]
def main():
    # Example job description
    jd = """We are looking for a Data Analyst with strong skills in SQL, Python,
            data visualization, and reporting. Experience with business intelligence
            tools like Power BI or Tableau is preferred."""
    
    resumes = load_resumes("data/processed")
    ranked = rank_resumes(jd, resumes, top_n=10)
    
    # Save results to CSV
    df = pd.DataFrame(ranked)
    df.to_csv("data/ranked_results.csv", index=False)
    
    print("Top candidates:")
    print(df[["filename", "score"]])

if __name__ == "__main__":
    main()
