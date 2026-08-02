"""
Day 2 - Document loaders (PDF, DOCX, URL, MD/TXT)
--------------------------------------------------
No API keys needed for the loaders themselves.

Shows:
1. Individual loaders for each format
2. LoaderRegistry pattern dispatching by extension
3. Light text cleaning

Run:
    python main.py
"""

import re
from pathlib import Path
from typing import Callable

import trafilatura
from docx import Document
from pypdf import PdfReader


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n\n".join(
        (p.extract_text() or "").strip() for p in reader.pages if (p.extract_text() or "").strip()
    )


def load_docx(path: str) -> str:
    doc = Document(path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_url(url: str) -> str:
    html = trafilatura.fetch_url(url)
    return trafilatura.extract(html) or "" if html else ""


LOADERS: dict[str, Callable[[str], str]] = {
    ".pdf":  load_pdf,
    ".docx": load_docx,
    ".md":   load_text,
    ".txt":  load_text,
}


def load(path: str) -> str:
    if path.startswith("http"):
        return load_url(path)
    ext = Path(path).suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"No loader registered for extension: {ext}")
    return loader(path)


def clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


if __name__ == "__main__":
    print("--- LoaderRegistry demo ---\n")
    print("Registered extensions:", list(LOADERS.keys()))
    demo_md = Path("README.md")
    if demo_md.exists():
        text = clean(load(str(demo_md)))
        print(f"\nLoaded README.md ({len(text)} chars). First 200:")
        print(text[:200])
    else:
        print("\nDrop a README.md next to this file to see it load.")
