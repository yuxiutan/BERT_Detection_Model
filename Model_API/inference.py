#!/usr/bin/env python3
"""
BERT 模型推理引擎
負責載入模型並執行預測
"""

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

class BERTDetectionInference:
    """BERT 檢測模型推理類"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.max_length = 512
        
        # 設定模型路徑優先順序
        if model_path:
            self.model_path = Path(model_path)
        else:
            # 優先使用當前模型，備選原始模型
            current_model = Path("/app/Model/bert_model_current.pth")
            original_model = Path("/app/Model/bert_model.pth")
            
            if current_model.exists():
                self.model_path = current_model
            elif original_model.exists():
                self.model_path = original_model
            else:
                raise FileNotFoundError("找不到模型檔案")
        
        # 載入模型
        self._load_model()
        
        logger.info(f"✅ BERT 推理引擎初始化完成，使用模型: {self.model_path}")
    
    def _load_model(self):
        """載入 BERT 模型和 tokenizer"""
        try:
            logger.info(f"📦 正在載入模型: {self.model_path}")
            
            # 載入 tokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            # 載入模型架構
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=2  # 二分類：0-正常, 1-攻擊
            )
            
            # 載入訓練好的權重
            if self.model_path.exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # 處理不同的保存格式
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)
                
                logger.info("✅ 模型權重載入成功")
            else:
                logger.warning("⚠️ 模型檔案不存在，使用預訓練權重")
            
            # 移到設備並設為評估模式
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"載入模型失敗: {str(e)}")
            raise
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        對輸入文本進行預測
        
        Args:
            text: 要檢測的日誌文本
            
        Returns:
            Tuple[str, float]: (預測結果, 信心度)
        """
        try:
            if not self.model or not self.tokenizer:
                raise RuntimeError("模型或 tokenizer 未正確載入")
            
            # 預處理文本
            inputs = self._preprocess_text(text)
            
            # 執行推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # 計算概率和預測
                probabilities = F.softmax(logits, dim=-1)
                confidence = torch.max(probabilities).item()
                predicted_class = torch.argmax(logits, dim=-1).item()
            
            # 將數值預測轉換為文字標籤
            prediction_label = "攻擊" if predicted_class == 1 else "正常"
            
            logger.debug(f"預測結果: {prediction_label}, 信心度: {confidence:.4f}")
            
            return prediction_label, confidence
            
        except Exception as e:
            logger.error(f"預測失敗: {str(e)}")
            return "錯誤", 0.0
    
    def _preprocess_text(self, text: str) -> dict:
        """預處理輸入文本"""
        try:
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
            
            # 移到設備
            return {
                'input_ids': encoding['input_ids'].to(self.device),
                'attention_mask': encoding['attention_mask'].to(self.device)
            }
            
        except Exception as e:
            logger.error(f"文本預處理失敗: {str(e)}")
            raise
    
    def batch_predict(self, texts: list) -> list:
        """
        批量預測
        
        Args:
            texts: 文本列表
            
        Returns:
            list: [(預測結果, 信心度), ...]
        """
        try:
            results = []
            
            for text in texts:
                prediction, confidence = self.predict(text)
                results.append((prediction, confidence))
            
            return results
            
        except Exception as e:
            logger.error(f"批量預測失敗: {str(e)}")
            return [("錯誤", 0.0)] * len(texts)
    
    def reload_model(self, new_model_path: Optional[str] = None):
        """重新載入模型（用於模型更新後）"""
        try:
            logger.info("🔄 重新載入模型...")
            
            if new_model_path:
                self.model_path = Path(new_model_path)
            
            # 重新載入模型
            self._load_model()
            
            logger.info("✅ 模型重新載入完成")
            
        except Exception as e:
            logger.error(f"重新載入模型失敗: {str(e)}")
            raise
    
    def get_model_info(self) -> dict:
        """獲取模型資訊"""
        try:
            info = {
                "model_path": str(self.model_path),
                "device": str(self.device),
                "max_length": self.max_length,
                "model_loaded": self.model is not None,
                "tokenizer_loaded": self.tokenizer is not None
            }
            
            if self.model_path.exists():
                stat = self.model_path.stat()
                info.update({
                    "model_size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "last_modified": stat.st_mtime
                })
            
            return info
            
        except Exception as e:
            logger.error(f"獲取模型資訊失敗: {str(e)}")
            return {"error": str(e)}
