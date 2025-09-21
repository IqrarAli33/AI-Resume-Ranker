import os, json, uuid, base64, mimetypes, secrets
from io import BytesIO
from pathlib import Path
import zipfile
import pandas as pd
import re

from flask import (
    Flask, render_template, request, redirect, url_for,
    send_from_directory, session, send_file, abort
)
from werkzeug.utils import secure_filename, safe_join
from werkzeug.exceptions import RequestEntityTooLarge

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TEMPLATE_DIR = (PROJECT_ROOT / "templates").resolve()
DATA_DIR = (PROJECT_ROOT / "data").resolve()
UPLOAD_FOLDER = DATA_DIR / "uploads"
TMP_FOLDER = DATA_DIR / "tmp"
FEEDBACK_FILE = DATA_DIR / "feedback.csv"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
TMP_FOLDER.mkdir(parents=True, exist_ok=True)
FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.secret_key = os.environ.get("APP_SECRET", secrets.token_hex(32))
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

ALLOWED_EXT = {".pdf", ".docx", ".doc"}

from bulk_parser import parse_resumes_concurrent
from parser_skills import extract_candidate_phrases
from ranker import rank_resumes
from retrain_model import retrain_model


def _allowed(fname: str) -> bool:
    return Path(fname).suffix.lower() in ALLOWED_EXT

def _save_results(ranked):
    file_id = str(uuid.uuid4())
    out = TMP_FOLDER / f"results_{file_id}.json"
    out.write_text(json.dumps(ranked, ensure_ascii=False), encoding="utf-8")
    session["results_file"] = str(out)
    return out

def _load_results():
    s = session.get("results_file")
    if not s:
        return None
    p = Path(s)
    if not p.exists() or not p.is_file():
        session.pop("results_file", None)
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        session.pop("results_file", None)
        return None

def _render_index_error(msg: str, code: int = 400):
    return render_template("index.html", error=msg), code

def _results_as_dataframe():
    ranked = _load_results()
    if not ranked:
        return None
    rows = []
    for i, r in enumerate(ranked, start=1):
        comp = r.get("components", {}) or {}
        rows.append({
            "Rank": i,
            "Filename": r.get("filename"),
            "Score": r.get("score"),
            "RawScore": r.get("raw_score", r.get("score")),
            "YearsExperience": r.get("years_experience", 0),
            "Shortlisted": bool(r.get("shortlisted", False)),
            "doc": comp.get("doc"),
            "bm25": comp.get("bm25"),
            "jacc": comp.get("jacc"),
            "fuzzy": comp.get("fuzzy"),
            "soft": comp.get("soft"),
            "para": comp.get("para"),
            "cover": comp.get("cover"),
            "ModelScore": r.get("model_score"),
        })
    return pd.DataFrame(rows)

# routes

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        jd_text = (request.form.get("jd_text") or "").strip()
        uploaded = request.files.getlist("resumes")

        if not jd_text or not uploaded:
            return _render_index_error("Please add a job description and at least one resume.", 400)

        resumes = parse_resumes_concurrent(UPLOAD_FOLDER, uploaded)
        if not resumes:
            return _render_index_error("No valid resumes found. Please upload PDF or Word files.", 400)

        jd_terms = extract_candidate_phrases(jd_text)
        ranked = rank_resumes(job_description=jd_text, resumes=resumes, jd_terms=jd_terms)

        _save_results(ranked)
        return render_template("results.html", ranked=ranked)

    if request.method == "GET" and request.args.get("fresh") == "1":
        return render_template("index.html")

    ranked = _load_results()
    if ranked:
        try:
            min_score = float(request.args.get("min_score", 0) or 0)
        except Exception:
            min_score = 0.0
        try:
            min_exp = int(request.args.get("min_exp", 0) or 0)
        except Exception:
            min_exp = 0

        keyword_raw = (request.args.get("keyword", "") or "").strip().lower()
        keywords = [t for t in re.split(r"[,\s]+", keyword_raw) if t]

        filtered = []
        for r in ranked:
            if float(r.get("score", 0)) < min_score:
                continue
            if int(r.get("years_experience", 0)) < min_exp:
                continue
            if keywords:
                text_lc = (r.get("text", "") or "").lower()
                if not all(k in text_lc for k in keywords):
                    continue
            filtered.append(r)
        return render_template("results.html", ranked=filtered)

    return render_template("index.html")

@app.post("/upload_zip")
def upload_zip():
    zfile = request.files.get("zipfile")
    jd_text = (request.form.get("jd_text") or "").strip()

    if not zfile or not jd_text:
        return _render_index_error("Upload a ZIP file and include a job description.", 400)

    data = BytesIO(zfile.read())
    try:
        with zipfile.ZipFile(data) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            extracted = []
            for n in names:
                fn = secure_filename(Path(n).name)
                if not fn:
                    continue
                if not _allowed(fn):
                    continue
                dest = (UPLOAD_FOLDER / fn)
                with zf.open(n) as src:
                    dest.write_bytes(src.read())
                extracted.append(dest)
    except zipfile.BadZipFile:
        return _render_index_error("Invalid ZIP file. Please try again.", 400)

    if not extracted:
        return _render_index_error("No supported resumes found in the ZIP.", 400)

    class _Shim:
        def __init__(self, p): self.filename = p.name; self._p = p
        def save(self, dst): (UPLOAD_FOLDER / self.filename).replace(dst)

    files = [_Shim(p) for p in extracted]
    resumes = parse_resumes_concurrent(UPLOAD_FOLDER, files)
    if not resumes:
        return _render_index_error("Could not process resumes from the ZIP.", 400)

    jd_terms = extract_candidate_phrases(jd_text)
    ranked = rank_resumes(job_description=jd_text, resumes=resumes, jd_terms=jd_terms)

    _save_results(ranked)
    return render_template("results.html", ranked=ranked)

@app.route("/download/<path:filename>", endpoint="download_resume")
def download_resume(filename):
    safe_name = secure_filename(filename)
    if not _allowed(safe_name):
        abort(400)
    return send_from_directory(
        UPLOAD_FOLDER, safe_name,
        as_attachment=True, download_name=safe_name
    )

@app.get("/preview_bin")
def preview_bin():
    token = request.args.get("id")
    if not token:
        abort(400)
    try:
        pad = "=" * (-len(token) % 4)
        fname = base64.urlsafe_b64decode(token + pad).decode("utf-8")
    except Exception:
        abort(400)

    safe_name = secure_filename(fname)
    full_path = safe_join(str(UPLOAD_FOLDER), safe_name)
    if not full_path or not os.path.isfile(full_path):
        abort(404)

    mime_type, _ = mimetypes.guess_type(full_path)
    resp = send_file(full_path,
                     mimetype=mime_type or "application/octet-stream",
                     as_attachment=False,
                     download_name=os.path.basename(full_path))
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.post("/feedback")
def feedback():
    decisions = []
    for key, value in request.form.items():
        if key.startswith("decision_") and value:
            filename = key.replace("decision_", "")
            decisions.append([filename, (value or "").strip().lower()])

    if decisions:
        df = pd.DataFrame(decisions, columns=["filename", "decision"])
        if FEEDBACK_FILE.exists():
            df.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
        else:
            df.to_csv(FEEDBACK_FILE, index=False)

    return redirect(url_for("history"))

@app.get("/results", endpoint="results")
def results():
    ranked = _load_results()
    if not ranked:
        try:
            latest = max(TMP_FOLDER.glob("results_*.json"),
                         key=lambda p: p.stat().st_mtime)
            ranked = json.loads(latest.read_text(encoding="utf-8"))
            session["results_file"] = str(latest)
        except Exception:
            ranked = None
    if ranked:
        return render_template("results.html", ranked=ranked)
    return render_template("index.html",
                           error="No previous results available. Please upload resumes and run ranking first.")

@app.get("/history")
def history():
    if FEEDBACK_FILE.exists():
        df = pd.read_csv(FEEDBACK_FILE)
        tables = [df.to_html(classes="table table-striped table-hover align-middle mb-0",
                             index=False, escape=False)]
    else:
        tables = ["<p>No feedback saved yet.</p>"]
    return render_template("history.html", tables=tables)

@app.get("/retrain")
def retrain():
    retrain_model()
    return "Model retrained successfully!"

@app.get("/export.csv")
def export_csv():
    df = _results_as_dataframe()
    if df is None or df.empty:
        abort(404)
    out = TMP_FOLDER / "ranked.csv"
    df.to_csv(out, index=False)
    return send_file(out, as_attachment=True,
                     download_name="ranked.csv", mimetype="text/csv")

@app.get("/export.xlsx")
def export_xlsx():
    df = _results_as_dataframe()
    if df is None or df.empty:
        abort(404)
    out = TMP_FOLDER / "ranked.xlsx"
    engine = None
    try:
        import xlsxwriter
        engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl
            engine = "openpyxl"
        except Exception:
            return _render_index_error("Please install xlsxwriter or openpyxl for Excel export.", 500)
    with pd.ExcelWriter(out, engine=engine) as writer:
        df.to_excel(writer, index=False, sheet_name="Ranked")
        try:
            ws = writer.sheets["Ranked"]
            for i, col in enumerate(df.columns, 1):
                max_len = max([len(str(col))] + [len(str(x)) for x in df[col].head(100).astype(str).tolist()])
                ws.set_column(i-1, i-1, min(max_len + 2, 40))
        except Exception:
            pass
    return send_file(out, as_attachment=True,
                     download_name="ranked.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@app.get("/export.pdf")
def export_pdf():
    df = _results_as_dataframe()
    if df is None or df.empty:
        abort(404)
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    except Exception:
        return _render_index_error("Please install reportlab to enable PDF export.", 501)
    out = TMP_FOLDER / "ranked.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=landscape(A4), leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    title = Paragraph("AI Resume Ranker — Ranked Results", styles["Heading2"])
    subtitle = Paragraph(f"Total: {len(df)}  •  Exported from current run", styles["Normal"])
    cols = ["Rank","Filename","Score","RawScore","YearsExperience","Shortlisted","doc","bm25","jacc","fuzzy","soft"]
    cols = [c for c in cols if c in df.columns]
    data = [cols] + df[cols].astype(object).values.tolist()
    from reportlab.platypus import Table
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#343a40")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 10),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.HexColor("#f7f7f7")]),
        ("FONTSIZE", (0,1), (-1,-1), 9),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story = [title, Spacer(1, 6), subtitle, Spacer(1, 12), table]
    doc.build(story)
    return send_file(out, as_attachment=True, download_name="ranked.pdf", mimetype="application/pdf")

@app.get("/reset")
def reset():
    session.pop("results_file", None)
    return redirect(url_for("index"))

@app.errorhandler(RequestEntityTooLarge)
def handle_413(e):
    return _render_index_error("Your upload is too large. Please use fewer or smaller files, or upload a ZIP.", 413)

if __name__ == "__main__":
    app.run(debug=True)
