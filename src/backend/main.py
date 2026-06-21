import langchain_text_splitters
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import spacy, xgboost as xgb, json
from transformers import BartForConditionalGeneration, BartTokenizer

from src.rag.rag_chain import run_rag
from src.ml.train_outcome import predict_outcome, build_features_single
from src.ml.train_ner import extract_entities
from src.ml.train_summarizer import summarize


models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    models["ner"] = spacy.load("models/legal_ner")
    models["xgb"] = xgb.XGBClassifier()
    models["xgb"].load_model("models/outcome_xgb.json")
    models["bart_tok"] = BartTokenizer.from_pretrained("models/legal_summarizer")
    models["bart"]     = BartForConditionalGeneration.from_pretrained("models/legal_summarizer")
    models["feat_cols"] = json.load(open("models/feature_cols.json"))
    print("All models loaded.")
    yield

app = FastAPI(title="LexMind API", version="1.0.0", lifespan=lifespan)


class CaseQuery(BaseModel):
    case_brief: str
    jurisdiction: str = "Supreme Court"
    year_from: int = 2000

class LexMindResponse(BaseModel):
    summary: str
    entities: dict
    precedents: list
    outcome: dict
    answer: str


@app.get("/")
async def health():
    return {"status": "ok", "models_loaded": list(models.keys())}


@app.post("/analyze", response_model=LexMindResponse)
async def analyze_case(query: CaseQuery):
    try:

        entities = extract_entities(query.case_brief, models["ner"])


        summary = summarize(
            query.case_brief,
            models["bart"], models["bart_tok"]
        )


        features = build_features_single(
            entities, query.jurisdiction, models["feat_cols"], query.case_brief
        )
        outcome = predict_outcome(features, models["xgb"], models["feat_cols"])


        rag_result = run_rag(query.case_brief, court=query.jurisdiction)

        return LexMindResponse(
            summary=summary,
            entities=entities,
            precedents=rag_result["sources"],
            outcome=outcome,
            answer=rag_result["answer"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
