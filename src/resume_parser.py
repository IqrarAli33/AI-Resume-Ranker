import os, re, tempfile, subprocess
import fitz  # PyMuPDF
from docx import Document

def _convert_doc_to_docx(src_path: str) -> str:
    """
    Optional: Convert legacy .doc → .docx using LibreOffice (if installed).
    If conversion fails, we return the original path and let preview/download handle it.
    """
    try:
        out_dir = tempfile.mkdtemp()
        subprocess.run(
            ["soffice", "--headless", "--convert-to", "docx", "--outdir", out_dir, src_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        base = os.path.basename(src_path)
        docx_path = os.path.join(out_dir, os.path.splitext(base)[0] + ".docx")
        if os.path.exists(docx_path):
            return docx_path
    except Exception:
        pass
    return src_path  # fallback

def extract_text(path):
    """Extract text from PDF, DOCX, or (optionally) DOC resumes."""
    path = str(path)
    low = path.lower()

    if low.endswith(".pdf"):
        text = ""
        with fitz.open(path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text

    # Try to convert .doc to .docx if possible; otherwise, we won't extract rich text
    if low.endswith(".doc"):
        path = _convert_doc_to_docx(path)
        low = path.lower()

    if low.endswith(".docx"):
        try:
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""

    return ""

def extract_years_of_experience(text: str) -> int:
    """
    Heuristic: returns the max number preceding 'years/yrs/yr'.
    Examples: '10+ years', '7 yrs', '3 yr'
    """
    text = (text or "").lower()
    matches = re.findall(r'(\d+)\s*\+?\s*(years|yrs|yr)\b', text)
    if matches:
        years = [int(m[0]) for m in matches]
        return max(years)
    return 0
