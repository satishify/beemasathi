"""FastAPI backend for BeemaSathi."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

from config import INSURERS
from rag.retriever import PolicyRetriever
from rag.generator import BimaSaathiGenerator
from agents.claim_predictor import ClaimPredictor

app = FastAPI(title="BeemaSathi API", version="0.1.0")

# Singletons
retriever = PolicyRetriever()
generator = BimaSaathiGenerator()
predictor = ClaimPredictor()


class QueryRequest(BaseModel):
    question: str
    insurer: Optional[str] = None
    policy_name: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    safe: bool
    sources: List[Dict]


class ClaimPredictRequest(BaseModel):
    insurer: str
    policy_name: str
    hospital_name: Optional[str] = None
    is_network_hospital: Optional[bool] = None
    procedure: Optional[str] = None
    room_rent_per_day: Optional[float] = None
    sum_insured: Optional[float] = None
    has_ped: bool = False
    months_since_policy: Optional[int] = None
    intimation_hours: Optional[int] = None


@app.get("/")
def root():
    return {"message": "BeemaSathi API is running", "version": "0.1.0"}


@app.get("/insurers")
def list_insurers():
    return INSURERS


@app.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    """RAG-based Q&A on policy documents."""
    try:
        chunks = retriever.retrieve(
            query=req.question,
            insurer=req.insurer,
            policy_name=req.policy_name,
            n_results=8
        )
        result = generator.generate(
            chunks,
            req.question,
            catalog=retriever.store.get_catalog(),
        )
        return QueryResponse(
            answer=result["answer"],
            safe=result["safe"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict_claim(req: ClaimPredictRequest):
    """Predict claim rejection risk."""
    try:
        result = predictor.predict(
            insurer=req.insurer,
            policy_name=req.policy_name,
            hospital_name=req.hospital_name,
            is_network_hospital=req.is_network_hospital,
            procedure=req.procedure,
            room_rent_per_day=req.room_rent_per_day,
            sum_insured=req.sum_insured,
            has_ped=req.has_ped,
            months_since_policy=req.months_since_policy,
            intimation_hours=req.intimation_hours,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "vector_db_chunks": retriever.store.get_stats()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
