# bulk_parser.py
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from werkzeug.utils import secure_filename

from resume_parser import extract_text, extract_years_of_experience

def _parse_one(upload_dir: Path, file_storage):
    filename = secure_filename(file_storage.filename or "")
    if not filename:
        return None
    save_path = upload_dir / filename
    file_storage.save(save_path)

    # extract text + heuristic years
    text = extract_text(save_path) or ""
    years = extract_years_of_experience(text)
    return {
        "filename": filename,
        "text": text,
        "years_experience": int(years),
        "score": 0.0,
        "path": str(save_path),
    }

def parse_resumes_concurrent(upload_dir: Path, file_storages, max_workers: int = None):
    """
    I/O-bound parsing in parallel. Safe for PDFs/DOCX; limits CPU contention.
    """
    items = [fs for fs in (file_storages or []) if getattr(fs, "filename", None)]
    if not items:
        return []

    max_workers = max_workers or min(32, (os.cpu_count() or 4) * 4)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_parse_one, upload_dir, fs) for fs in items]
        for fut in as_completed(futures):
            try:
                rec = fut.result()
                if rec:
                    results.append(rec)
            except Exception:
                # skip problematic file but keep going
                continue
    return results
