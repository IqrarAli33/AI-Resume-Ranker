from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from pathlib import Path
import pandas as pd
from werkzeug.utils import secure_filename
from resume_parser import extract_text, extract_years_of_experience
from ranker import rank_resumes
import json
app = Flask(__name__, template_folder="../templates")

UPLOAD_FOLDER = Path("data/uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)

FEEDBACK_FILE = Path("data/feedback.csv")
FEEDBACK_FILE.parent.mkdir(exist_ok=True, parents=True)


def process_uploaded_resumes(files):
    resumes = []
    for f in files:
        filename = secure_filename(f.filename)
        save_path = UPLOAD_FOLDER / filename
        f.save(save_path)

        text = extract_text(save_path)
        years_exp = extract_years_of_experience(text)  # ✅ now cleanly imported

        resumes.append({
            "filename": filename,
            "raw_text": text,
            "years_experience": years_exp,
            "score": 0.0,
            "path": str(save_path)
        })
    return resumes

# -----------------------------
# Routes
# -----------------------------
app.secret_key = "supersecret"   # required for session

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        jd_text = request.form.get("jd_text", "")
        uploaded_resumes = request.files.getlist("resumes")

        if not jd_text.strip() or not uploaded_resumes:
            return render_template("index.html", error="Please upload resumes and provide a job description.")

        # Process resumes
        resumes = process_uploaded_resumes(uploaded_resumes)

        # Rank resumes
        ranked = rank_resumes(jd_text, resumes)

        # ✅ Store results in session (as JSON)
        session["ranked"] = json.dumps(ranked)

        return render_template("results.html", jd=jd_text, ranked=ranked)

    # Handle filters when GET request with ranked results in session
    if request.method == "GET" and "ranked" in session:
        ranked = json.loads(session["ranked"])

        # Apply filters
        min_score = float(request.args.get("min_score", 0))
        min_exp = int(request.args.get("min_exp", 0))
        keyword = request.args.get("keyword", "").lower()

        filtered = [
            r for r in ranked
            if r["score"] >= min_score
            and r["years_experience"] >= min_exp
            and (keyword in r["text"].lower() if keyword else True)
        ]

        return render_template("results.html", jd="", ranked=filtered)

    return render_template("index.html")

@app.route("/download/<filename>")
def download_resume(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)


@app.route("/feedback", methods=["POST"])
def feedback():
    decisions = []
    for key, value in request.form.items():
        if key.startswith("decision_") and value:
            filename = key.replace("decision_", "")
            decisions.append([filename, value])

    if decisions:
        df = pd.DataFrame(decisions, columns=["filename", "decision"])
        if FEEDBACK_FILE.exists():
            df.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
        else:
            df.to_csv(FEEDBACK_FILE, index=False)

    return redirect(url_for("history"))


@app.route("/history")
def history():
    if FEEDBACK_FILE.exists():
        df = pd.read_csv(FEEDBACK_FILE)
        tables = [df.to_html(classes="table table-striped", index=False)]
    else:
        tables = ["<p>No feedback yet.</p>"]
    return render_template("history.html", tables=tables)


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
