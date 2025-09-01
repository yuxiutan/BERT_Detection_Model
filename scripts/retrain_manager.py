#!/usr/bin/env python3
"""
重訓練管理器
負責自動化模型重訓練流程，包含模型備份、版本管理和自動化訓練
"""

import os
import json
import shutil
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
import torch

from Model_API.retrain_model import ModelRetrainer

logger = logging.getLogger(__name__)

class RetrainManager:
    def __init__(self):
        self.model_dir = Path("/app/Model")
        self.backup_dir = Path("/app/model_backups")
        self.low_confidence_file = Path("/app/low_confidence_data/low_confidence_events.json")
        self.data_dir = Path("/app/data")
        
        # 模型檔案路徑
        self.original_model_path = self.model_dir / "bert_model.pth"
        self.current_model_path = self.model_dir / "bert_model_current.pth"
        
        # 備份設定
        self.max_backups = int(os.getenv('MODEL_BACKUP_COUNT', '3'))
        
        # 確保目錄存在
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型檔案
        self._initialize_model_files()
        
        logger.info(f"🔄 重訓練管理器初始化完成，最大備份數: {self.max_backups}")
    
    def _initialize_model_files(self):
        """初始化模型檔案，確保有原始權重檔案"""
        try:
            # 如果沒有原始模型檔案，創建一個
            if not self.original_model_path.exists():
                logger.warning("⚠️ 未找到原始模型檔案，請確保模型檔案存在")
                return
            
            # 如果沒有當前模型檔案，從原始模型複製
            if not self.current_model_path.exists():
                shutil.copy2(self.original_model_path, self.current_model_path)
                logger.info("📋 已從原始模型創建當前模型檔案")
                
        except Exception as e:
            logger.error(f"初始化模型檔案失敗: {str(e)}")
    
    async def retrain_model(self) -> bool:
        """
        執行模型重訓練流程
        
        Returns:
            bool: 重訓練是否成功
        """
        try:
            logger.info("🚀 開始自動化模型重訓練流程...")
            
            # 1. 檢查低信心度資料
            low_confidence_data = self._load_low_confidence_data()
            if not low_confidence_data:
                logger.info("📭 沒有低信心度資料，跳過重訓練")
                return True
            
            logger.info(f"📊 找到 {len(low_confidence_data)} 個低信心度事件用於重訓練")
            
            # 2. 備份當前模型
            backup_success = self._backup_current_model()
            if not backup_success:
                logger.error("❌ 模型備份失敗，中止重訓練")
                return False
            
            # 3. 準備訓練資料
            training_data_path = self._prepare_training_data(low_confidence_data)
            if not training_data_path:
                logger.error("❌ 準備訓練資料失敗")
                return False
            
            # 4. 執行重訓練
            retrain_success = await self._execute_retrain(training_data_path)
            if not retrain_success:
                logger.error("❌ 重訓練執行失敗")
                return False
            
            # 5. 驗證新模型
            validation_success = self._validate_new_model()
            if not validation_success:
                logger.error("❌ 新模型驗證失敗，恢復備份")
                self._restore_from_backup()
                return False
            
            # 6. 清理低信心度資料檔案
            self._clear_low_confidence_data()
            
            # 7. 清理舊備份
            self._cleanup_old_backups()
            
            logger.info("✅ 模型重訓練流程完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 重訓練流程失敗: {str(e)}")
            return False
    
    def _load_low_confidence_data(self) -> List[Dict[str, Any]]:
        """載入低信心度事件資料"""
        try:
            if not self.low_confidence_file.exists():
                return []
            
            events = []
            with open(self.low_confidence_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
            
            return events
        except Exception as e:
            logger.error(f"載入低信心度資料失敗: {str(e)}")
            return []
    
    def _backup_current_model(self) -> bool:
        """備份當前模型"""
        try:
            if not self.current_model_path.exists():
                logger.warning("
