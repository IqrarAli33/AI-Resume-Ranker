from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rank_resumes(job_description, resumes):
    docs = [job_description] + [r["raw_text"] for r in resumes]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs)
    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]
    scores = cosine_similarity(jd_vector, resume_vectors)[0]

    for i, r in enumerate(resumes):
        r["score"] = round(float(scores[i]), 4)

    ranked = sorted(resumes, key=lambda x: x["score"], reverse=True)
    return ranked
