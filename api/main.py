"""FastAPI backend for BeemaSathi."""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

from api.rate_limit import TokenBucketLimiter
from config import INSURERS
from agents.claim_predictor import ClaimPredictor
from agents.orchestrator import BeemaSathiOrchestrator

app = FastAPI(title="BeemaSathi API", version="0.2.0")

LIMITER = TokenBucketLimiter(
    rate_per_minute=int(os.getenv("BEEMASATHI_RATE_LIMIT", "30")),
    burst=int(os.getenv("BEEMASATHI_RATE_BURST", "10")),
)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path in {"/", "/health"}:
        return await call_next(request)
    client = request.client.host if request.client else "anonymous"
    allowed, retry_after = LIMITER.check(client)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Retry shortly."},
            headers={"Retry-After": str(retry_after)},
        )
    return await call_next(request)

orchestrator = BeemaSathiOrchestrator()
retriever = orchestrator.retriever
generator = orchestrator.generator
predictor = orchestrator.predictor


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    insurer: Optional[str] = None
    policy_name: Optional[str] = None
    strategy: Optional[str] = "baseline"
    claim: Optional[Dict] = None


class QueryResponse(BaseModel):
    answer: str
    safe: bool
    sources: List[Dict]
    plan: Optional[Dict] = None
    verification: Optional[Dict] = None


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
    return {"message": "BeemaSathi API is running", "version": "0.2.0"}


@app.get("/insurers")
def list_insurers():
    return INSURERS


@app.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    """Agentic RAG: plan, retrieve, answer, verify."""
    try:
        result = orchestrator.answer(
            question=req.question,
            insurer=req.insurer,
            policy_name=req.policy_name,
            strategy=req.strategy or "baseline",
            claim=req.claim,
        )
        return QueryResponse(
            answer=result["answer"],
            safe=result["safe"],
            sources=result["sources"],
            plan=result.get("plan"),
            verification=result.get("verification"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plan")
def plan_question(req: QueryRequest):
    """Return the typed task plan without calling the model."""
    plan = orchestrator.route(req.question, req.insurer, req.policy_name)
    return plan.as_dict()


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
