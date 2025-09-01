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
                logger.warning("⚠️ 當前模型檔案不存在，無法備份")
                return False
            
            # 生成備份檔案名稱（包含時間戳）
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_filename = f"bert_model_backup_{timestamp}.pth"
            backup_path = self.backup_dir / backup_filename
            
            # 執行備份
            shutil.copy2(self.current_model_path, backup_path)
            logger.info(f"💾 模型已備份至: {backup_filename}")
            
            return True
            
        except Exception as e:
            logger.error(f"模型備份失敗: {str(e)}")
            return False
    
    def _prepare_training_data(self, low_confidence_events: List[Dict[str, Any]]) -> str:
        """準備重訓練資料"""
        try:
            # 創建訓練資料檔案
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            training_file = self.data_dir / f"retrain_data_{timestamp}.json"
            
            # 轉換資料格式供訓練使用
            training_data = []
            for event in low_confidence_events:
                training_sample = {
                    "text": event.get("log_data", ""),
                    "label": 1,  # 假設低信心度事件需要重新標記為攻擊
                    "timestamp": event.get("timestamp"),
                    "confidence": event.get("confidence", 0.0),
                    "agent_name": event.get("agent_name"),
                    "rule_id": event.get("rule_id")
                }
                training_data.append(training_sample)
            
            # 寫入檔案
            with open(training_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📝 已準備 {len(training_data)} 筆重訓練資料: {training_file.name}")
            return str(training_file)
            
        except Exception as e:
            logger.error(f"準備訓練資料失敗: {str(e)}")
            return None
    
    async def _execute_retrain(self, training_data_path: str) -> bool:
        """執行模型重訓練"""
        try:
            logger.info("🔥 開始執行模型重訓練...")
            
            # 初始化重訓練器
            retrainer = ModelRetrainer(
                model_path=str(self.current_model_path),
                training_data_path=training_data_path,
                output_path=str(self.current_model_path)
            )
            
            # 執行重訓練（這裡應該是異步的）
            success = await retrainer.retrain()
            
            if success:
                logger.info("✅ 模型重訓練完成")
                return True
            else:
                logger.error("❌ 模型重訓練失敗")
                return False
                
        except Exception as e:
            logger.error(f"執行重訓練失敗: {str(e)}")
            return False
    
    def _validate_new_model(self) -> bool:
        """驗證新訓練的模型"""
        try:
            logger.info("🔍 驗證新訓練的模型...")
            
            # 檢查模型檔案是否存在且可載入
            if not self.current_model_path.exists():
                logger.error("新模型檔案不存在")
                return False
            
            # 嘗試載入模型
            try:
                model_state = torch.load(self.current_model_path, map_location='cpu')
                logger.info("✅ 新模型檔案驗證成功")
                return True
            except Exception as e:
                logger.error(f"新模型載入失敗: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"模型驗證失敗: {str(e)}")
            return False
    
    def _restore_from_backup(self) -> bool:
        """從最新備份恢復模型"""
        try:
            logger.info("🔄 正在從備份恢復模型...")
            
            # 找到最新的備份檔案
            backup_files = list(self.backup_dir.glob("bert_model_backup_*.pth"))
            if not backup_files:
                logger.error("沒有找到備份檔案")
                return False
            
            # 按檔名排序，取最新的
            latest_backup = sorted(backup_files)[-1]
            
            # 恢復備份
            shutil.copy2(latest_backup, self.current_model_path)
            logger.info(f"✅ 已從備份恢復: {latest_backup.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"從備份恢復失敗: {str(e)}")
            return False
    
    def _clear_low_confidence_data(self):
        """清空低信心度事件資料檔案"""
        try:
            if self.low_confidence_file.exists():
                # 備份舊資料
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                backup_file = self.low_confidence_file.parent / f"processed_low_confidence_{timestamp}.json"
                shutil.move(self.low_confidence_file, backup_file)
                logger.info(f"📦 低信心度資料已備份至: {backup_file.name}")
            
            # 創建新的空檔案
            self.low_confidence_file.touch()
            
        except Exception as e:
            logger.error(f"清理低信心度資料失敗: {str(e)}")
    
    def _cleanup_old_backups(self):
        """清理過舊的模型備份，只保留最新的 N 個"""
        try:
            backup_files = list(self.backup_dir.glob("bert_model_backup_*.pth"))
            
            if len(backup_files) <= self.max_backups:
                return
            
            # 按檔名排序（包含時間戳）
            backup_files.sort()
            
            # 刪除多餘的舊備份
            files_to_delete = backup_files[:-self.max_backups]
            
            for file_path in files_to_delete:
                file_path.unlink()
                logger.info(f"🗑️ 已刪除舊備份: {file_path.name}")
            
            logger.info(f"✅ 備份清理完成，保留最新 {self.max_backups} 個備份")
            
        except Exception as e:
            logger.error(f"清理備份失敗: {str(e)}")
    
    def get_current_model_version(self) -> str:
        """獲取當前模型版本資訊"""
        try:
            if self.current_model_path.exists():
                mtime = self.current_model_path.stat().st_mtime
                return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            else:
                return "未知"
        except Exception as e:
            logger.error(f"獲取模型版本失敗: {str(e)}")
            return "錯誤"
    
    def get_backup_list(self) -> List[Dict[str, Any]]:
        """獲取備份模型列表"""
        try:
            backup_files = list(self.backup_dir.glob("bert_model_backup_*.pth"))
            backup_files.sort(reverse=True)  # 最新的在前
            
            backups = []
            for backup_file in backup_files:
                stat = backup_file.stat()
                backups.append({
                    "filename": backup_file.name,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created_time": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
                    "modified_time": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                })
            
            return backups
            
        except Exception as e:
            logger.error(f"獲取備份列表失敗: {str(e)}")
            return []
    
    def restore_from_backup(self, backup_filename: str) -> bool:
        """從指定備份恢復模型"""
        try:
            backup_path = self.backup_dir / backup_filename
            
            if not backup_path.exists():
                logger.error(f"備份檔案不存在: {backup_filename}")
                return False
            
            # 先備份當前模型
            current_backup_success = self._backup_current_model()
            if not current_backup_success:
                logger.warning("⚠️ 當前模型備份失敗，但繼續恢復操作")
            
            # 恢復指定備份
            shutil.copy2(backup_path, self.current_model_path)
            logger.info(f"✅ 已從備份恢復模型: {backup_filename}")
            
            return True
            
        except Exception as e:
            logger.error(f"從備份恢復失敗: {str(e)}")
            return False
