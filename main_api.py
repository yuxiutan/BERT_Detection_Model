#!/usr/bin/env python3
"""
BERT Detection Model API with Swagger UI
主要的 FastAPI 應用程式，包含模型推理、定時檢測和 Swagger UI
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch

from Model_API.inference import BERTDetectionInference
from utils.wazuh_api import WazuhAPIClient
from scripts.retrain_manager import RetrainManager

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class DetectionRequest(BaseModel):
    log_data: str
    agent_name: Optional[str] = None
    rule_id: Optional[str] = None

class DetectionResponse(BaseModel):
    prediction: str
    confidence: float
    timestamp: str
    is_low_confidence: bool

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    last_detection_time: Optional[str]
    total_detections: int
    low_confidence_count: int

# 創建 FastAPI 應用
app = FastAPI(
    title="BERT Detection Model API",
    description="自動化網路安全事件檢測系統，包含定時檢測、自動重訓練和低信心度事件處理",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全域變量
detection_model = None
wazuh_client = None
retrain_manager = None
detection_stats = {
    "total_detections": 0,
    "low_confidence_count": 0,
    "last_detection_time": None
}

# 環境變數
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.3'))
DETECTION_INTERVAL = int(os.getenv('DETECTION_INTERVAL', '180'))  # 3分鐘
LOW_CONFIDENCE_FILE = "/app/low_confidence_data/low_confidence_events.json"

@app.on_event("startup")
async def startup_event():
    """應用啟動時初始化模型和服務"""
    global detection_model, wazuh_client, retrain_manager
    
    try:
        logger.info("🚀 正在啟動 BERT Detection API...")
        
        # 初始化檢測模型
        logger.info("📦 正在載入檢測模型...")
        detection_model = BERTDetectionInference()
        
        # 初始化 Wazuh API 客戶端
        logger.info("🔗 正在初始化 Wazuh API 客戶端...")
        wazuh_client = WazuhAPIClient()
        
        # 初始化重訓練管理器
        logger.info("🔄 正在初始化重訓練管理器...")
        retrain_manager = RetrainManager()
        
        # 創建低信心度事件檔案目錄
        os.makedirs(os.path.dirname(LOW_CONFIDENCE_FILE), exist_ok=True)
        
        # 啟動定時檢測背景任務
        asyncio.create_task(periodic_detection_task())
        
        logger.info("✅ BERT Detection API 啟動完成！")
        
    except Exception as e:
        logger.error(f"❌ 啟動失敗: {str(e)}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康檢查端點"""
    return HealthResponse(
        status="healthy" if detection_model else "unhealthy",
        model_loaded=detection_model is not None,
        last_detection_time=detection_stats["last_detection_time"],
        total_detections=detection_stats["total_detections"],
        low_confidence_count=detection_stats["low_confidence_count"]
    )

@app.post("/detect", response_model=DetectionResponse)
async def detect_single_event(request: DetectionRequest):
    """單一事件檢測端點"""
    if not detection_model:
        raise HTTPException(status_code=503, detail="模型尚未載入")
    
    try:
        # 執行檢測
        prediction, confidence = detection_model.predict(request.log_data)
        
        # 更新統計
        detection_stats["total_detections"] += 1
        detection_stats["last_detection_time"] = datetime.now(timezone.utc).isoformat()
        
        is_low_confidence = confidence < CONFIDENCE_THRESHOLD
        
        # 如果信心度低，保存到檔案
        if is_low_confidence:
            await save_low_confidence_event({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "log_data": request.log_data,
                "agent_name": request.agent_name,
                "rule_id": request.rule_id,
                "prediction": prediction,
                "confidence": confidence
            })
            detection_stats["low_confidence_count"] += 1
        
        return DetectionResponse(
            prediction=prediction,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_low_confidence=is_low_confidence
        )
        
    except Exception as e:
        logger.error(f"檢測失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"檢測失敗: {str(e)}")

@app.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """手動觸發重訓練"""
    if not retrain_manager:
        raise HTTPException(status_code=503, detail="重訓練管理器尚未初始化")
    
    background_tasks.add_task(retrain_manager.retrain_model)
    return {"message": "重訓練任務已開始執行"}

@app.get("/stats")
async def get_detection_stats():
    """獲取檢測統計資訊"""
    return {
        **detection_stats,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "detection_interval_seconds": DETECTION_INTERVAL,
        "model_version": retrain_manager.get_current_model_version() if retrain_manager else "unknown"
    }

@app.get("/low-confidence-events")
async def get_low_confidence_events(limit: int = 100):
    """獲取低信心度事件列表"""
    try:
        if not os.path.exists(LOW_CONFIDENCE_FILE):
            return {"events": [], "total": 0}
        
        with open(LOW_CONFIDENCE_FILE, 'r', encoding='utf-8') as f:
            events = [json.loads(line) for line in f if line.strip()]
        
        # 按時間倒序排列，返回最近的事件
        events.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return {
            "events": events[:limit],
            "total": len(events)
        }
    except Exception as e:
        logger.error(f"讀取低信心度事件失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"讀取失敗: {str(e)}")

async def save_low_confidence_event(event_data: Dict[str, Any]):
    """保存低信心度事件到 JSON 檔案"""
    try:
        with open(LOW_CONFIDENCE_FILE, 'a', encoding='utf-8') as f:
            json.dump(event_data, f, ensure_ascii=False)
            f.write('\n')
        logger.info(f"已保存低信心度事件: {event_data['timestamp']}")
    except Exception as e:
        logger.error(f"保存低信心度事件失敗: {str(e)}")

async def periodic_detection_task():
    """定時檢測背景任務 - 每3分鐘執行一次"""
    logger.info(f"🕒 定時檢測任務已啟動，間隔 {DETECTION_INTERVAL} 秒")
    
    while True:
        try:
            await asyncio.sleep(DETECTION_INTERVAL)
            
            if not wazuh_client or not detection_model:
                logger.warning("⚠️ Wazuh 客戶端或模型尚未初始化，跳過此次檢測")
                continue
            
            logger.info("🔍 開始定時檢測...")
            
            # 從 Wazuh 獲取最新資料
            recent_events = await wazuh_client.get_recent_events(minutes=3)
            
            if not recent_events:
                logger.info("📭 沒有新的事件需要檢測")
                continue
            
            logger.info(f"📊 檢測到 {len(recent_events)} 個新事件")
            
            # 批量檢測
            low_confidence_events = []
            
            for event in recent_events:
                try:
                    # 構造檢測請求
                    log_data = event.get('full_log', '')
                    if not log_data:
                        continue
                    
                    # 執行檢測
                    prediction, confidence = detection_model.predict(log_data)
                    
                    # 更新統計
                    detection_stats["total_detections"] += 1
                    detection_stats["last_detection_time"] = datetime.now(timezone.utc).isoformat()
                    
                    # 檢查是否為低信心度
                    if confidence < CONFIDENCE_THRESHOLD:
                        low_confidence_event = {
                            "timestamp": event.get('timestamp', datetime.now(timezone.utc).isoformat()),
                            "log_data": log_data,
                            "agent_name": event.get('agent.name'),
                            "rule_id": event.get('rule.id'),
                            "prediction": prediction,
                            "confidence": confidence,
                            "source": "periodic_detection"
                        }
                        low_confidence_events.append(low_confidence_event)
                        
                except Exception as e:
                    logger.error(f"檢測單一事件失敗: {str(e)}")
                    continue
            
            # 批量保存低信心度事件
            if low_confidence_events:
                for event in low_confidence_events:
                    await save_low_confidence_event(event)
                detection_stats["low_confidence_count"] += len(low_confidence_events)
                logger.info(f"💾 已保存 {len(low_confidence_events)} 個低信心度事件")
            
            logger.info(f"✅ 定時檢測完成，處理了 {len(recent_events)} 個事件")
            
        except Exception as e:
            logger.error(f"❌ 定時檢測任務失敗: {str(e)}")

if __name__ == "__main__":
    # 開發模式運行
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
