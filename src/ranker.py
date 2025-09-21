# src/ranker.py
"""
Batch-adaptive resume ranking:
- Stage A (all): TF-IDF cosine + BM25 (both 0..1 after scaling)
- Stage B (top-K): jaccard/fuzzy/soft on phrases, paragraph max-cosine, JD key-term coverage, exp-fit
- Optional Learning-to-Rank blending from data/weights.json (trained by retrain_model.py)
- Final per-batch min-max scaling -> best ≈ 1.000, interpretable spread for recruiters
"""

import re
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from parser_skills import extract_candidate_phrases   # your helper
from match_skills import jaccard, fuzzy_cover, soft_tfidf_sim  # your helper

# --------- JD focusing so fluff ("About/Culture") doesn't dilute relevance ----------
SEC_HEADINGS = (
    "requirements", "required", "qualifications", "technical skills",
    "preferred", "nice to have", "must have", "must-have",
    "responsibilities", "skills", "experience"
)

def _focus_jd(text: str) -> str:
    if not text:
        return ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    keep, take = [], False
    for ln in lines:
        low = ln.lower()
        if any(h in low for h in SEC_HEADINGS):
            take = True
            keep.append(ln)
            continue
        if take:
            keep.append(ln)
    focused = "\n".join(keep)
    return focused if len(focused) > 200 else text


# --------------------- utilities ---------------------
def _minmax(x: np.ndarray) -> np.ndarray:
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


# ---- BM25 over unigrams (batch-adaptive, fast) ----
def _bm25_scores(resume_texts, jd_text, max_features=60000, k1=1.5, b=0.75):
    """
    BM25(doc, query) where query is the JD. No domain lists; uses batch vocabulary.
    Uses safe conversions (no .A1 on sparse matrices).
    Returns 0..1 scaled scores for readability.
    """
    vec = CountVectorizer(stop_words="english", max_features=max_features)
    X = vec.fit_transform(resume_texts)  # CSR (N_docs, V)
    if X.shape[1] == 0:
        return np.zeros(X.shape[0], dtype=np.float32)

    # Query term counts (only terms in vocab matter)
    q = vec.transform([jd_text])         # CSR (1, V)
    q_idx = q.nonzero()[1]
    if q_idx.size == 0:
        return np.zeros(X.shape[0], dtype=np.float32)

    # Document frequency and idf per BM25
    df_mat = (X > 0).sum(axis=0)                       # 1 x V (matrix)
    df = np.asarray(df_mat).ravel()                    # -> (V,)
    N = X.shape[0]
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)    # vector (V,)

    # Length normalization
    dl_mat = X.sum(axis=1)                             # N x 1 (matrix)
    dl = np.asarray(dl_mat).ravel().astype(np.float32) # -> (N,)
    avgdl = float(dl.mean()) if N > 0 else 0.0
    denom_const = (k1 * (1.0 - b)) + (k1 * b * (dl / (avgdl + 1e-9)))

    scores = np.zeros(N, dtype=np.float32)
    for t in q_idx:
        tf = X[:, t].toarray().ravel().astype(np.float32)     # safe across SciPy versions
        num = tf * (k1 + 1.0)
        denom = tf + denom_const
        scores += (idf[t] * (num / (denom + 1e-9))).astype(np.float32)

    # rescale 0..1 for readability
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)


# ---- Paragraph max cosine: best paragraph should match the JD ----
_para_split = re.compile(r"\n\s*\n+")  # paragraphs = blocks separated by blank lines

def _max_paragraph_cosine(jd_vec, vec, resume_text, max_paragraphs=50):
    """
    Compare JD vector to each paragraph of the resume and take the max cosine.
    Helps when only a section of the resume matches the JD strongly.
    """
    text = (resume_text or "").strip()
    parts = _para_split.split(text)[:max_paragraphs] if text else []
    if not parts:
        # fallback: sentence-ish chunks if the resume has no blank lines
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)[:max_paragraphs]
    try:
        P = vec.transform(parts)           # reuse the same TF-IDF vectorizer
        cs = cosine_similarity(jd_vec, P).flatten()
        return float(cs.max()) if cs.size else 0.0
    except Exception:
        return 0.0


def _top_jd_terms(tfidf_vec, jd_row, top_k=25):
    """Pick top-k weighted terms from the JD vector (auto 'must-haves')."""
    try:
        feats = tfidf_vec.get_feature_names_out()
        row = jd_row.toarray().ravel()
        top_idx = np.argsort(-row)[:top_k]
        return [feats[i] for i in top_idx if row[i] > 0]
    except Exception:
        return []


def _coverage_ratio(text, terms, use_fuzzy=True):
    """What fraction of JD key terms appear in the resume (substring/fuzzy)? 0..1"""
    if not terms:
        return 0.0
    low = (text or "").lower()
    hits = 0
    for t in terms:
        t = (t or "").lower().strip()
        if not t:
            continue
        ok = (t in low)
        if not ok and use_fuzzy:
            try:
                from rapidfuzz import fuzz
                ok = fuzz.partial_ratio(t, low) >= 85
            except Exception:
                ok = False
        if ok:
            hits += 1
    return hits / max(len(terms), 1)


# --------------------- Learning-to-Rank blending ---------------------
L2R_FEATURES = ["doc","bm25","jacc","fuzzy","soft","para","cover","raw_score","years_norm"]

def _load_ltr_weights():
    """Load weights produced by retrain_model.py, or return None."""
    try:
        p = Path(__file__).parent / "data" / "weights.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None

def _features_for_model(r):
    c = r.get("components", {}) or {}
    years = float(r.get("years_experience", 0) or 0.0)
    years_norm = min(years / 40.0, 1.0)
    raw = float(r.get("raw_score", r.get("score", 0.0)))
    return [
        float(c.get("doc", 0.0)),
        float(c.get("bm25", 0.0)),
        float(c.get("jacc", 0.0)),
        float(c.get("fuzzy", 0.0)),
        float(c.get("soft", 0.0)),
        float(c.get("para", 0.0)),
        float(c.get("cover", 0.0)),
        float(raw),
        float(years_norm),
    ]


# --------------------------- main API ---------------------------
def rank_resumes(
    job_description,
    resumes,
    jd_terms=None,
    jd_min_exp=0,
    min_shortlist=0.65,          # used if shortlist_by == "threshold"
    max_features=40000,
    ngram_range=(1, 2),
    top_k_deep=150,
    dtype=np.float32,
    rescale="minmax",            # "minmax" or "none"
    shortlist_by="percentile",   # "percentile" or "threshold"
    shortlist_cut=0.80           # top 20% shortlisted when percentile mode
):
    """
    Returns the same list of resume dicts with:
      - score (float)
      - shortlisted (bool)
      - components: dict of signal values
      - raw_score (pre-calibration score) for audits/training
      - model_score (if weights.json is present)
    """
    if not resumes:
        return []

    # ---- focus JD ----
    jd_text = _focus_jd(job_description or "")

    # ---- Stage A: TF-IDF doc cosine + BM25 (fast for all) ----
    docs = [jd_text] + [r.get("text", "") for r in resumes]
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
        dtype=dtype,
        sublinear_tf=True
    )
    X = tfidf.fit_transform(docs)                 # (1 + N, V)
    jd_vec = X[0:1]
    R_vecs = X[1:]
    doc_cos = cosine_similarity(jd_vec, R_vecs).flatten().astype(np.float32)
    if rescale == "minmax":
        doc_cos = _minmax(doc_cos)

    bm25 = _bm25_scores([r.get("text", "") for r in resumes], jd_text).astype(np.float32)
    base_stageA = (0.6 * doc_cos + 0.4 * bm25).astype(np.float32)  # both already 0..1

    # seed scores with Stage A and set components
    for i, r in enumerate(resumes):
        r["score"] = float(base_stageA[i])
        r["shortlisted"] = False
        r["components"] = {"doc": round(float(doc_cos[i]), 6), "bm25": round(float(bm25[i]), 6)}

    # pick top-K for deeper work
    order = np.argsort(-base_stageA)
    deep_idx = set(order[:min(top_k_deep, len(resumes))])

    # JD terms once (domain-agnostic phrases)
    jd_terms = set(jd_terms or extract_candidate_phrases(jd_text))
    # auto key terms from JD tf-idf (e.g., python/sql/regression/… depending on JD)
    jd_keyterms = _top_jd_terms(tfidf, jd_vec, top_k=25)

    # ---- Stage B: only for top-K ----
    for i, r in enumerate(resumes):
        if i not in deep_idx:
            continue

        text = r.get("text", "")
        cv_terms = set(extract_candidate_phrases(text))
        j = jaccard(jd_terms, cv_terms)
        fz = fuzzy_cover(jd_terms, cv_terms)
        st = soft_tfidf_sim(jd_terms, cv_terms)
        para = _max_paragraph_cosine(jd_vec, tfidf, text)
        cover = _coverage_ratio(text, jd_keyterms, use_fuzzy=True)
        exp_fit = 1.0 if r.get("years_experience", 0) >= jd_min_exp else 0.6

        # blend: baseA + deep components (sum to 1.0)
        base = float(base_stageA[i])
        score = (
            0.50 * base +
            0.15 * j +
            0.10 * fz +
            0.10 * st +
            0.10 * para +
            0.05 * cover
        )
        score = score * (0.9 + 0.1 * exp_fit)

        r["score"] = float(score)
        r["components"].update({
            "jacc": round(j, 6),
            "fuzzy": round(fz, 6),
            "soft": round(st, 6),
            "para": round(para, 6),
            "cover": round(cover, 6),
            "exp_fit": round(exp_fit, 2),
        })

    # ---- Save raw (pre-calibration) final score for audits/training ----
    for r in resumes:
        r["raw_score"] = round(float(r["score"]), 6)

    # ---- Optional: blend with learned model (if available) BEFORE min-max ----
    W = _load_ltr_weights()
    if W and W.get("features") == L2R_FEATURES and "coef" in W:
        coef = np.array(W["coef"], dtype=np.float64)
        intercept = float(W.get("intercept", 0.0))
        for r in resumes:
            x = np.array(_features_for_model(r), dtype=np.float64)
            z = float(np.dot(coef, x) + intercept)
            prob = 1.0 / (1.0 + np.exp(-z))  # logistic
            r["model_score"] = round(prob, 6)
            # Blend: 50% current score, 50% learned probability
            r["score"] = 0.5 * float(r["score"]) + 0.5 * prob

    # ---- Batch calibration of FINAL scores ----
    scores = np.array([float(r["score"]) for r in resumes], dtype=np.float32)
    if rescale == "minmax":
        scaled = _minmax(scores)
        for i, r in enumerate(resumes):
            r["score"] = round(float(scaled[i]), 6)
    else:
        for r in resumes:
            r["score"] = round(float(r["score"]), 6)

    # ---- Shortlist rule ----
    if shortlist_by == "percentile":
        thr = float(np.quantile([r["score"] for r in resumes], shortlist_cut))
        for r in resumes:
            r["shortlisted"] = r["score"] >= thr
    else:
        for r in resumes:
            r["shortlisted"] = r["score"] >= min_shortlist

    resumes.sort(key=lambda x: x["score"], reverse=True)
    return resumes
