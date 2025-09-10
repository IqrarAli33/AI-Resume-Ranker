import re
import fitz  # PyMuPDF
from docx import Document

def extract_text(path):
    """Extract text from PDF or DOCX resumes."""
    path = str(path)
    if path.endswith(".pdf"):
        text = ""
        with fitz.open(path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text
    elif path.endswith(".docx"):
        doc = Document(path)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

def extract_years_of_experience(text: str) -> int:
    """
    Simple heuristic to extract years of experience from resume text.
    Looks for patterns like '5 years', '10+ years', etc.
    """
    matches = re.findall(r'(\d+)\+?\s+years', text.lower())
    if matches:
        years = [int(m) for m in matches]
        return max(years)  # assume highest number mentioned
    return 0
