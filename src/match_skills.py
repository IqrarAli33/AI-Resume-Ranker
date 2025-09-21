from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from rapidfuzz import fuzz
    _FUZZ = True
except Exception:
    _FUZZ = False

def jaccard(a:set, b:set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0

def fuzzy_cover(jd_terms, cv_terms):
    """
    Average best fuzzy match from each JD term into the resume terms.
    Returns 0..1. Uses rapidfuzz if available; otherwise returns 0.
    """
    if not _FUZZ or not jd_terms or not cv_terms:
        return 0.0
    scores = []
    cv_list = list(cv_terms)
    for j in jd_terms:
        best = 0
        for c in cv_list:
            sc = fuzz.partial_ratio(j, c)
            if sc > best:
                best = sc
        scores.append(best / 100.0)
    return sum(scores)/len(scores) if scores else 0.0

def soft_tfidf_sim(jd_terms, cv_terms):
    """
    Soft similarity over the *sets* of phrases (1–2gram TF-IDF).
    """
    docs = ["; ".join(jd_terms or []), "; ".join(cv_terms or [])]
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english', min_df=1)
    X = vec.fit_transform(docs)
    return float(cosine_similarity(X[0:1], X[1:2])[0,0])
