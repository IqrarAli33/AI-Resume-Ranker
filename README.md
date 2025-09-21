# AI Resume Ranker

A fast, job-adaptive resume ranking web app. Paste a Job Description (JD), upload up to **500+ resumes** (PDF/DOCX or a ZIP), and get an interpretable, ranked shortlist in seconds. Includes recruiter feedback and an offline **retrain** step to keep improving results.

## ✨ Features

- **Batch-adaptive ranking**  
  Stage-A (all resumes): TF-IDF cosine + **BM25**  
  Stage-B (top-K): Jaccard / fuzzy / soft-TFIDF on phrases, **paragraph max-similarity**, JD key-term coverage, experience fit.
- **Calibrated scores (0..1)**  
  Per-run min–max scaling: best candidate is ~**1.000**; easy to read and compare within a batch.
- **Learning-to-Rank (optional)**  
  `/retrain` learns weights from recruiter **feedback.csv** and blends a model probability into the final score.
- **Lovely UI**  
  Dark mode, drag-and-drop upload, progress hints, powerful filters, inline **PDF/DOCX preview**, secure downloads.
- **Exports**  
  One-click **CSV / XLSX / PDF** exports of the current run.
- **Scales to 500+** resumes  
  Concurrent parsing + two-stage scoring keep it fast for large batches.

---

## 🗂️ Project structure

```
project/
├─ src/
│  ├─ app.py                    # Flask app (routes, exports, session cache)
│  ├─ ranker.py                 # Ranking engine (Stage-A/Stage-B + L2R blending)
│  ├─ bulk_parser.py            # Parallel parsing of many resumes (PDF/DOCX/ZIP)
│  ├─ resume_parser.py          # Text + years-of-experience extraction
│  ├─ parser_skills.py          # Phrase extraction helpers (JD & resumes)
│  ├─ match_skills.py           # jaccard/fuzzy/soft TFIDF utilities
│  ├─ retrain_model.py          # Trains LogisticRegression → data/weights.json
│  └─ data/
│     ├─ uploads/               # Uploaded files (per run)
│     ├─ tmp/                   # Cached run results (results_*.json), exports
│     ├─ feedback.csv           # Recruiter labels (selected/rejected)
│     └─ weights.json           # Learned weights (created by /retrain)
├─ templates/
│  ├─ index.html                # JD + upload UI
│  ├─ results.html              # Ranked table, filters, preview, export buttons
│  └─ history.html              # Feedback history dashboard
├─ requirements.txt
└─ README.md
```

---

## 🧰 Requirements

- **Python** 3.10+ (3.12 works great)
- pip packages (install below):
  - `flask`, `pandas`, `numpy`, `scikit-learn`, `scipy`
  - **parsers**: `PyMuPDF` (a.k.a. `pymupdf`), `python-docx`
  - **optional**: `rapidfuzz` (better fuzzy coverage)
  - **export**: `xlsxwriter` *or* `openpyxl` (for XLSX), `reportlab` (for PDF)
- Frontend uses Bootstrap + Mammoth (DOCX preview) via CDN (already wired in templates)

---

## ⚙️ Setup

```bash
# 1) Create & activate a venv
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
# If you don't have a requirements.txt yet:
pip install flask pandas numpy scikit-learn scipy pymupdf python-docx rapidfuzz xlsxwriter openpyxl reportlab
```

> **Tip (Windows)**: If a package build fails, upgrade `pip setuptools wheel` and try again.

---

## 🚀 Run

```bash
# from project root
python src/app.py
```

Open: http://127.0.0.1:5000

**Flow**
1. Paste a **Job Description**.
2. Upload many **PDF/DOCX** files or a **ZIP** of resumes.
3. Click **Rank resumes** → see ranked table.
4. Preview / Download individual resumes.
5. Save **Selected / Rejected** → builds `data/feedback.csv`.
6. Export **CSV / XLSX / PDF** (buttons on Results page).

---

## 🧠 Retraining (Learning-to-Rank)

Once you have some feedback:

1) From Results, choose **Selected/Rejected** and click **Save Feedback**.  
2) Hit the retrain endpoint:

```
GET /retrain
```

This reads `data/tmp/results_*.json` (features) + `data/feedback.csv` (labels), trains a small **LogisticRegression**, and writes **`data/weights.json`**.  
On the **next run**, `ranker.py` auto-loads these weights and blends the model probability into the final score.

> Not enough feedback yet? The trainer writes safe defaults; the system falls back to the strong heuristic blend.

---

## 📤 Exports

Routes (also available via buttons in `results.html`):

- `GET /export.csv`   → `ranked.csv`
- `GET /export.xlsx`  → `ranked.xlsx` (requires `xlsxwriter` or `openpyxl`)
- `GET /export.pdf`   → `ranked.pdf` (requires `reportlab`)

---

## 🔩 Configuration knobs

You can tweak these in `ranker.rank_resumes(...)` (defaults are sensible):

- `top_k_deep=150` — deep features only for top-K (raise for accuracy, lower for speed).
- `shortlist_by="percentile", shortlist_cut=0.80` — shortlist top 20% (set `0.90` for top 10%).
- `max_features=40000` — vocab cap for TF-IDF/BM25 (20k–60k common).
- `rescale="minmax"` — keeps scores in 0..1 per run. Use `raw_score` if you need cross-run comparisons.

---

## 🔒 Security notes (minimum)

- Filenames are sanitized; uploads stored under `data/uploads/`.
- For production, add:
  - Authentication (Flask-Login), roles (Recruiter/Admin)
  - HTTPS + secure cookies
  - Size limits & rate limiting
  - Storage retention policy and, if needed, encryption at rest

---

## ⚡ Performance tips

- 200–500 resumes: should be smooth on a modern laptop.
- If latency grows:
  - reduce `top_k_deep` (e.g., 150 → 120 or 100),
  - lower `max_features` (e.g., 40k → 30k),
  - ensure venv runs with optimized SciPy/NumPy,
  - consider running on a machine with more RAM/CPU cores.

---

## 🛠️ Troubleshooting

- **Template not found**: start the app from project root so Flask can find `templates/`.
- **Large ZIP fails**: increase `MAX_CONTENT_LENGTH` in `app.py`.
- **XLSX/PDF export errors**:  
  `pip install xlsxwriter` *or* `pip install openpyxl`; for PDF: `pip install reportlab`.
- **Scores look tiny (e.g., 0.2)**: that’s the *raw* cosine look; final **calibrated** score maps the best to ~1.000. Use the UI’s `Score` column (calibrated) for ranking.

---

## 🧪 What the score means

- `Score` (0..1): **calibrated** per batch — best ≈ 1.000 this run.
- `RawScore`: pre-calibration final score (useful for audits / cross-run comparison).
- `ModelScore`: probability from the learned model (after you run `/retrain`).
- `Shortlisted`: true/false (by percentile or threshold).

---

## 🤝 Contributing

PRs welcome. Please open issues for bugs or feature requests.

