import requests, json, time, os
from dotenv import load_dotenv


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

API_KEY = os.getenv("INDIAN_KANOON_API_KEY")
BASE_URL = "https://api.indiankanoon.org"

def search_cases(query: str, pages: int = 5) -> list:
    """Search and return case documents from Indian Kanoon."""
    if not API_KEY:
        print("ERROR: INDIAN_KANOON_API_KEY not found in environment variables.")
        return []
        
    headers = {"Authorization": f"Token {API_KEY}"}
    all_docs = []

    for page in range(pages):
        print(f"  Searching page {page} for query: '{query}'...")

        resp = requests.post(
            f"{BASE_URL}/search/",
            data={"formInput": query, "pagenum": page},
            headers=headers, timeout=10
        )
        if resp.status_code != 200:
            print(f"  Error page {page}: {resp.status_code}")
            break
        data = resp.json()
        docs = data.get("docs", [])
        if not docs:
            break
        all_docs.extend(docs)
        time.sleep(0.5)

    return all_docs

def fetch_full_doc(doc_id: str) -> dict:
    """Fetch full text of a single judgment."""
    if not API_KEY:
        return {}
        
    headers = {"Authorization": f"Token {API_KEY}"}
    resp = requests.post(
        f"{BASE_URL}/doc/{doc_id}/",
        headers=headers, timeout=15
    )
    return resp.json() if resp.status_code == 200 else {}

if __name__ == "__main__":
    queries = [
        "criminal appeal supreme court section 302",
        "civil breach of contract high court",
        "cheating fraud section 420 IPC",
        "dowry death section 304B IPC",
        "bail application criminal case",
    ]


    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    for q in queries:
        print(f"Scraping: {q}")
        docs = search_cases(q, pages=2)
        full_docs = []
        

        for doc in docs[:5]:
            doc_id = doc.get("tid")
            if doc_id:
                print(f"    Fetching full text for doc ID: {doc_id}...")
                full_text_data = fetch_full_doc(str(doc_id))
                if full_text_data:
                    doc["doc"] = full_text_data.get("doc", "")
                    full_docs.append(doc)
                time.sleep(0.5)
                
        fname = q.replace(" ", "_")[:40]
        output_path = os.path.join(raw_dir, f"{fname}.json")
        with open(output_path, "w") as f:
            json.dump(full_docs, f, indent=2)
        print(f"  Saved {len(full_docs)} docs with text to {output_path}")
