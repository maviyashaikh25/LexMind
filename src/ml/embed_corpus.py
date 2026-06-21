import json, os
import torch
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

MODEL_NAME = "law-ai/InLegalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

INDEX_NAME = os.getenv("PINECONE_ENV", "lexmind-cases")

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeddings(texts, batch_size=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        with torch.no_grad():
            model_output = model(**encoded_input)
            
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        all_embeddings.extend(batch_embeddings.cpu().numpy().tolist())
        
    return all_embeddings

def upsert_chunks(chunks_path: str, batch_size: int = 100):
    chunks = []
    with open(chunks_path) as f:
        chunks = [json.loads(l) for l in f]

    print(f"Embedding and upserting {len(chunks)} chunks...")
    for i in tqdm(range(0, len(chunks), batch_size), desc="Upserting"):
        batch = chunks[i:i+batch_size]
        texts   = [c["text"] for c in batch]
        vectors = get_embeddings(texts, batch_size=len(texts))

        upsert_data = [
            (
                c["chunk_id"],
                vectors[j],
                {
                    "doc_id": c["doc_id"],
                    "court": c.get("court", ""),
                    "year": c.get("year", ""),
                    "citation": c.get("citation", ""),
                    "text": c["text"][:500],
                }
            )
            for j, c in enumerate(batch)
        ]
        index.upsert(vectors=upsert_data)

    print(f"Indexed {len(chunks)} chunks into Pinecone.")

if __name__ == "__main__":
    upsert_chunks("data/processed/chunks.jsonl")