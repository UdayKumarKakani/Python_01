"""Ingest a PDF into the Pinecone knowledge base index (created if it doesn't exist).

Run from the project root:
    python -m scripts.ingest_pdf knowledge_base/acme_knowledge_base.pdf
    python -m scripts.ingest_pdf path/to/other.pdf --chunk-size 800 --chunk-overlap 120
    python -m scripts.ingest_pdf knowledge_base/acme_knowledge_base.pdf --dry-run   # parse + chunk only

Re-running on the same file replaces its previous chunks (vector IDs are derived
from the file name, page and chunk number).
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import get_settings
from app.llm import get_embedding_model
from app.vectorstore import ensure_index, get_index

EMBED_BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 100


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _normalize_line(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def _repeated_lines(page_texts: list[str]) -> set[str]:
    """Lines (digits ignored) that appear on most pages are running headers/footers."""
    if len(page_texts) < 3:
        return set()
    counts: dict[str, int] = {}
    for text in page_texts:
        for line in {_normalize_line(line) for line in text.splitlines() if line.strip()}:
            counts[line] = counts.get(line, 0) + 1
    return {line for line, n in counts.items() if n >= 0.6 * len(page_texts)}


def _clean(text: str, boilerplate: set[str]) -> str:
    lines = [line for line in text.splitlines() if _normalize_line(line) not in boilerplate]
    text = re.sub(r"[ \t]+", " ", "\n".join(lines))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_chunks(pdf_path: Path, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    doc_id = _slug(pdf_path.stem)
    page_texts = [page.extract_text() or "" for page in reader.pages]
    boilerplate = _repeated_lines(page_texts)
    chunks = []
    for page_number, raw_text in enumerate(page_texts, start=1):
        text = _clean(raw_text, boilerplate)
        for i, chunk in enumerate(splitter.split_text(text)):
            chunks.append(
                {
                    "id": f"{doc_id}-p{page_number:03d}-c{i:03d}",
                    "text": chunk,
                    "metadata": {"text": chunk, "source": pdf_path.name, "page": page_number, "chunk": i},
                }
            )
    return chunks


def _delete_previous(index: Any, doc_id: str, namespace: str | None) -> int:
    """Remove vectors from an earlier ingestion of the same document (serverless indexes)."""
    kwargs = {"prefix": f"{doc_id}-p", **({"namespace": namespace} if namespace else {})}
    deleted = 0
    try:
        for id_batch in index.list(**kwargs):
            if id_batch:
                index.delete(ids=list(id_batch), **({"namespace": namespace} if namespace else {}))
                deleted += len(id_batch)
    except Exception as exc:  # list() is unavailable on pod-based indexes
        print(f"  warning: could not remove previous chunks ({exc}); IDs will be overwritten in place.")
    return deleted


async def ingest(pdf_path: Path, chunk_size: int, chunk_overlap: int, dry_run: bool) -> None:
    settings = get_settings()
    chunks = load_chunks(pdf_path, chunk_size, chunk_overlap)
    if not chunks:
        raise SystemExit(f"No extractable text in {pdf_path}. Is it a scanned PDF? Run OCR first.")

    pages = len({c["metadata"]["page"] for c in chunks})
    print(f"Parsed {pdf_path.name}: {pages} pages -> {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    if dry_run:
        for c in chunks[:3]:
            print(f"\n--- {c['id']} ---\n{c['text'][:400]}")
        print("\nDry run: nothing was embedded or uploaded.")
        return

    print(f"Ensuring Pinecone index '{settings.pinecone_index_name}' exists...")
    await asyncio.to_thread(ensure_index, settings)
    index = get_index()
    namespace = settings.pinecone_namespace

    embedder = get_embedding_model()
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        vectors += await embedder.aembed_documents([c["text"] for c in batch])
        print(f"  embedded {min(start + EMBED_BATCH_SIZE, len(chunks))}/{len(chunks)}")

    records = [
        {"id": c["id"], "values": v, "metadata": c["metadata"]} for c, v in zip(chunks, vectors, strict=True)
    ]

    # Delete only after embedding succeeded, so a failed run leaves the old chunks in place.
    deleted = await asyncio.to_thread(_delete_previous, index, _slug(pdf_path.stem), namespace)
    if deleted:
        print(f"Removed {deleted} chunks from a previous ingestion.")
    for start in range(0, len(records), UPSERT_BATCH_SIZE):
        await asyncio.to_thread(
            index.upsert,
            vectors=records[start : start + UPSERT_BATCH_SIZE],
            **({"namespace": namespace} if namespace else {}),
        )
    print(f"Upserted {len(records)} chunks into '{settings.pinecone_index_name}'"
          + (f" (namespace '{namespace}')" if namespace else "") + ".")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path, help="Path to the PDF to ingest")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Max characters per chunk (default 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Characters shared by adjacent chunks")
    parser.add_argument("--dry-run", action="store_true", help="Parse and chunk only; skip Pinecone and embeddings")
    args = parser.parse_args()

    if not args.pdf.is_file() or args.pdf.suffix.lower() != ".pdf":
        sys.exit(f"Not a PDF file: {args.pdf}")
    if args.chunk_overlap >= args.chunk_size:
        sys.exit("--chunk-overlap must be smaller than --chunk-size")
    asyncio.run(ingest(args.pdf, args.chunk_size, args.chunk_overlap, args.dry_run))


if __name__ == "__main__":
    main()
