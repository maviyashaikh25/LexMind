import re, json, os, pandas as pd
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
METADATA_CSV = BASE_DIR / "data" / "cases_metadata.csv"
CHUNKS_JSONL = PROCESSED_DIR / "chunks.jsonl"


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?i)IN THE.{0,60}COURT', '', text)
    return text.strip()


OUTCOME_MAP = {
    "appeal allowed": 1,
    "petition allowed": 1,
    "appeal dismissed": 0,
    "petition dismissed": 0,
    "partly allowed": 2,
    "disposed of": 2,
}

def extract_outcome(text: str) -> int:
    text_lower = text.lower()[-500:]
    for phrase, label in OUTCOME_MAP.items():
        if phrase in text_lower:
            return label
    return -1


def extract_metadata(doc: dict) -> dict:
    publish_date = doc.get("publishdate") or "2000"
    return {
        "doc_id": doc.get("tid", "") or "",
        "title": doc.get("title", "") or "",
        "court": doc.get("docsource", "") or "",
        "date": doc.get("publishdate", "") or "",
        "citation": doc.get("citation", "") or "",
        "text_length": len(doc.get("doc", "") or ""),
        "year": str(publish_date)[:4],
    }


splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

def chunk_document(doc_id: str, text: str, metadata: dict) -> list:
    chunks = splitter.split_text(text)

    chunk_meta = {k: v for k, v in metadata.items() if k != "cleaned_text"}
    return [
        {"chunk_id": f"{doc_id}_c{i}",
         "text": c,
         "doc_id": doc_id,
         **chunk_meta}
        for i, c in enumerate(chunks)
    ]


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks, metadata_rows = [], []

    print(f"Scanning raw files in: {RAW_DIR}")
    for fpath in RAW_DIR.glob("*.json"):
        print(f"Processing {fpath.name}...")
        with open(fpath, encoding="utf-8") as f:
            try:
                docs = json.load(f)
            except Exception as e:
                print(f"Error loading {fpath.name}: {e}")
                continue

        for doc in docs:
            raw_text = doc.get("doc", "")
            if not raw_text or len(raw_text) < 200: 
                continue

            clean = clean_text(raw_text)
            meta  = extract_metadata(doc)
            meta["outcome"] = extract_outcome(clean)
            meta["cleaned_text"] = clean

            chunks = chunk_document(meta["doc_id"], clean, meta)
            all_chunks.extend(chunks)
            metadata_rows.append(meta)

    if metadata_rows:
        pd.DataFrame(metadata_rows).to_csv(METADATA_CSV, index=False)
        print(f"Saved metadata to {METADATA_CSV}")
        
        with open(CHUNKS_JSONL, "w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c) + "\n")
        print(f"Saved chunks to {CHUNKS_JSONL}")
        print(f"Pipeline complete: {len(metadata_rows)} cases -> {len(all_chunks)} chunks")
    else:
        print("No raw cases found to preprocess. Please run scraper.py first.")
