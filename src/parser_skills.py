import re
from collections import Counter
from itertools import chain

# NLTK with graceful fallback (auto-download lightweight corpora)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    _NLTK_OK = True
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        # Some environments need explicit wordnet download
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
except Exception:
    _NLTK_OK = False

_WORD = re.compile(r"[A-Za-z][A-Za-z\-\+/&\.]*")
STOP = set(stopwords.words("english")) if _NLTK_OK else set()
LEM = WordNetLemmatizer() if _NLTK_OK else None

def _maybe_lemma(tok: str) -> str:
    if LEM:
        try:
            return LEM.lemmatize(tok)
        except Exception:
            return tok
    return tok

def _tokens(text: str):
    for w in _WORD.findall((text or "").lower()):
        if w not in STOP and len(w) > 1:
            yield _maybe_lemma(w)

def _n_grams(tokens, n):
    toks = list(tokens)
    for i in range(len(toks)-n+1):
        yield " ".join(toks[i:i+n])

def _candidate_phrases(text: str):
    toks = list(_tokens(text))
    grams = chain.from_iterable(_n_grams(toks, n) for n in (1,2,3,4))
    grams = [g for g in grams if 2 <= len(g) <= 60]
    freq = Counter(grams)
    # Top 200, then prune substrings
    top = [p for p, _ in freq.most_common(200)]
    pruned = []
    for c in sorted(top, key=len):
        if not any(c in longer for longer in pruned):
            pruned.append(c)
    return pruned[:200]

def extract_candidate_phrases(text: str):
    """Domain-agnostic candidate phrases from any JD/resume text."""
    return _candidate_phrases(text or "")
