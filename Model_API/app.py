from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from Model_API.inference import run_inference
from Model_API.model_build import load_model

app = FastAPI(title="BERT Detection API", description="API for real-time attack detection", version="1.0")

class Event(BaseModel):
    timestamp: str
    agent_ip: Optional[str] = None
    agent_name: Optional[str] = None
    agent_id: Optional[str] = None
    rule_id: Optional[str] = None
    rule_mitre_id: Optional[str] = "T0000"
    rule_level: Optional[int] = None
    rule_description: Optional[str] = None
    data_srcip: Optional[str] = None
    data_dstip: Optional[str] = None
    full_log: str

model = load_model()  # 啟動時載入最新模型

@app.post("/predict", summary="Run inference on a single event")
def predict(event: Event):
    """對單一事件進行推論"""
    confidence = run_inference(model, event.dict())
    return {"confidence": confidence}

@app.get("/health", summary="Check API health")
def health():
    """檢查 API 狀態"""
    return {"status": "healthy"}
