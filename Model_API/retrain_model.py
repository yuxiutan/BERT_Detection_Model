#!/usr/bin/env python3
"""
模型重訓練器
負責使用低信心度資料進行增量訓練
"""

import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Dict, Any, Tuple
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

class RetrainDataset(Dataset):
    """重訓練資料集"""
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = str(item.get('text', ''))
        label = int(item.get('label', 0))
        
        # 編碼文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ModelRetrainer:
    """模型重訓練器"""
    
    def __init__(self, model_path: str, training_data_path: str, output_path: str):
        self.model_path = Path(model_path)
        self.training_data_path = Path(training_data_path)
        self.output_path = Path(output_path)
        
        # 訓練參數
        self.batch_size = 8
        self.learning_rate = 2e-5
        self.epochs = 3
        self.max_length = 512
        self.warmup_steps = 100
        
        # 設備
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️ 使用設備: {self.device}")
        
        # 初始化 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    async def retrain(self) -> bool:
        """執行重訓練流程"""
        try:
            logger.info("🔥 開始模型重訓練...")
            
            # 1. 載入訓練資料
            training_data = self._load_training_data()
            if not training_data:
                logger.error("❌ 沒有訓練資料")
                return False
            
            # 2. 載入現有模型
            model = self._load_existing_model()
            if model is None:
                logger.error("❌ 載入模型失敗")
                return False
            
            # 3. 準備資料載入器
            train_loader = self._prepare_dataloader(training_data)
            
            # 4. 設定優化器和排程器
            optimizer, scheduler = self._setup_optimizer_scheduler(model, train_loader)
            
            # 5. 執行訓練
            training_success = await self._train_model(model, train_loader, optimizer, scheduler)
            if not training_success:
                return False
            
            # 6. 保存模型
            save_success = self._save_model(model)
            if not save_success:
                return False
            
            logger.info("✅ 模型重訓練完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 重訓練失敗: {str(e)}")
            return False
    
    def _load_training_data(self) -> List[Dict[str, Any]]:
        """載入訓練資料"""
        try:
            with open(self.training_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"📊 載入 {len(data)} 筆訓練資料")
            return data
            
        except Exception as e:
            logger.error(f"載入訓練資料失敗: {str(e)}")
            return []
    
    def _load_existing_model(self) -> BertForSequenceClassification:
        """載入現有模型"""
        try:
            # 載入模型架構
            model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=2  # 二分類：正常/攻擊
            )
            
            # 載入權重
            if self.model_path.exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(checkpoint)
                logger.info(f"✅ 已載入現有模型權重: {self.model_path}")
            else:
                logger.warning("⚠️ 模型檔案不存在，使用預訓練權重")
            
            model.to(self.device)
            return model
            
        except Exception as e:
            logger.error(f"載入模型失敗: {str(e)}")
            return None
    
    def _prepare_dataloader(self, training_data: List[Dict[str, Any]]) -> DataLoader:
        """準備資料載入器"""
        dataset = RetrainDataset(training_data, self.tokenizer, self.max_length)
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Docker 環境中避免多進程問題
        )
    
    def _setup_optimizer_scheduler(self, model: nn.Module, train_loader: DataLoader) -> Tuple[AdamW, Any]:
        """設定優化器和學習率排程器"""
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        return optimizer, scheduler
    
    async def _train_model(self, model: nn.Module, train_loader: DataLoader, 
                          optimizer: AdamW, scheduler: Any) -> bool:
        """執行模型訓練"""
        try:
            model.train()
            
            for epoch in range(self.epochs):
                logger.info(f"📚 開始第 {epoch + 1}/{self.epochs} 輪訓練...")
                
                epoch_loss = 0
                predictions = []
                true_labels = []
                
                for batch_idx, batch in enumerate(train_loader):
                    # 移到設備
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # 前向傳播
                    optimizer.zero_grad()
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    epoch_loss += loss.item()
                    
                    # 反向傳播
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    # 收集預測結果用於評估
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)
                    predictions.extend(preds.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())
                    
                    # 每10批次記錄一次進度
                    if (batch_idx + 1) % 10 == 0:
                        logger.info(f"  批次 {batch_idx + 1}/{len(train_loader)}, 損失: {loss.item():.4f}")
                    
                    # 讓其他協程有機會執行
                    if batch_idx % 5 == 0:
                        await asyncio.sleep(0.01)
                
                # 計算輪次統計
                avg_loss = epoch_loss / len(train_loader)
                accuracy = accuracy_score(true_labels, predictions)
                
                logger.info(f"📊 第 {epoch + 1} 輪完成 - 平均損失: {avg_loss:.4f}, 準確率: {accuracy:.4f}")
            
            logger.info("✅ 模型訓練完成")
            return True
            
        except Exception as e:
            logger.error(f"訓練失敗: {str(e)}")
            return False
    
    def _save_model(self, model: nn.Module) -> bool:
        """保存訓練完成的模型"""
        try:
            # 確保輸出目錄存在
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存模型權重
            torch.save(model.state_dict(), self.output_path)
            
            # 保存模型資訊
            model_info = {
                "retrain_time": datetime.now(timezone.utc).isoformat(),
                "model_path": str(self.output_path),
                "training_data_path": str(self.training_data_path),
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size
            }
            
            info_path = self.output_path.parent / "model_info.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 模型已保存至: {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存模型失敗: {str(e)}")
            return False
