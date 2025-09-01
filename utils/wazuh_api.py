#!/usr/bin/env python3
"""
Wazuh API 客戶端
用於從 Wazuh 獲取最新的安全事件資料
"""

import os
import json
import aiohttp
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class WazuhAPIClient:
    def __init__(self):
        self.api_url = os.getenv('WAZUH_API_URL')
        self.username = os.getenv('WAZUH_API_USERNAME')
        self.password = os.getenv('WAZUH_API_PASSWORD')
        
        if not all([self.api_url, self.username, self.password]):
            raise ValueError("缺少必要的 Wazuh API 環境變數")
        
        self.session = None
        self.target_agents = [
            "DESKTOP-66TG6SE",
            "DESKTOP-66G2GGG", 
            "connector-node"
        ]
    
    async def __aenter__(self):
        """異步上下文管理器進入"""
        self.session = aiohttp.ClientSession(
            auth=aiohttp.BasicAuth(self.username, self.password),
            connector=aiohttp.TCPConnector(ssl=False),
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """異步上下文管理器退出"""
        if self.session:
            await self.session.close()
    
    async def get_recent_events(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """
        獲取最近指定分鐘數的事件
        
        Args:
            minutes: 獲取最近幾分鐘的資料
            
        Returns:
            List[Dict]: 事件列表
        """
        try:
            if not self.session:
                async with self:
                    return await self._fetch_events(minutes)
            else:
                return await self._fetch_events(minutes)
                
        except Exception as e:
            logger.error(f"獲取 Wazuh 事件失敗: {str(e)}")
            return []
    
    async def _fetch_events(self, minutes: int) -> List[Dict[str, Any]]:
        """內部方法：實際獲取事件"""
        events = []
        search_after = None
        total_fetched = 0
        
        logger.info(f"📡 開始獲取最近 {minutes} 分鐘的事件...")
        
        while True:
            # 構建查詢
            query = self._build_query(minutes, search_after)
            
            try:
                async with self.session.post(
                    self.api_url,
                    json=query,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    
                    if response.status != 200:
                        logger.error(f"API 請求失敗，狀態碼: {response.status}")
                        break
                    
                    data = await response.json()
                    hits = data.get('hits', {}).get('hits', [])
                    
                    if not hits:
                        break
                    
                    # 處理事件資料
                    batch_events = []
                    for hit in hits:
                        source = hit.get('_source', {})
                        event = self._process_event_data(source)
                        if event:
                            batch_events.append(event)
                    
                    events.extend(batch_events)
                    total_fetched += len(hits)
                    
                    logger.info(f"已獲取 {len(hits)} 筆，累計 {total_fetched} 筆...")
                    
                    # 設定下一次查詢的 search_after
                    if len(hits) < 10000:  # 如果返回的資料少於 size，表示已經是最後一頁
                        break
                    
                    search_after = hits[-1].get('sort')
                    
            except Exception as e:
                logger.error(f"API 請求異常: {str(e)}")
                break
        
        logger.info(f"✅ 獲取完成，總共 {len(events)} 筆有效事件")
        return events
    
    def _build_query(self, minutes: int, search_after: Optional[List] = None) -> Dict[str, Any]:
        """構建 Elasticsearch 查詢"""
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "timestamp": {
                                    "gte": f"now-{minutes}m",
                                    "lte": "now"
                                }
                            }
                        },
                        {
                            "terms": {
                                "agent.name": self.target_agents
                            }
                        }
                    ]
                }
            },
            "_source": [
                "timestamp",
                "agent.ip",
                "agent.name", 
                "agent.id",
                "rule.id",
                "rule.mitre.id",
                "rule.level",
                "rule.description",
                "data.srcip",
                "data.dstip",
                "full_log"
            ],
            "sort": [
                {
                    "timestamp": {
                        "order": "asc"
                    }
                }
            ],
            "size": 10000
        }
        
        if search_after:
            query["search_after"] = search_after
        
        return query
    
    def _process_event_data(self, source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """處理單一事件資料"""
        try:
            # 檢查是否有必要欄位
            if not source.get('full_log'):
                return None
            
            return {
                "timestamp": source.get('timestamp'),
                "agent.ip": source.get('agent', {}).get('ip'),
                "agent.name": source.get('agent', {}).get('name'),
                "agent.id": source.get('agent', {}).get('id'),
                "rule.id": source.get('rule', {}).get('id'),
                "rule.mitre.id": source.get('rule', {}).get('mitre', {}).get('id', 'T0000'),
                "rule.level": source.get('rule', {}).get('level'),
                "rule.description": source.get('rule', {}).get('description'),
                "data.srcip": source.get('data', {}).get('srcip'),
                "data.dstip": source.get('data', {}).get('dstip'),
                "full_log": source.get('full_log')
            }
        except Exception as e:
            logger.error(f"處理事件資料失敗: {str(e)}")
            return None

# 創建全域客戶端實例（用於非 FastAPI 上下文）
async def create_wazuh_client() -> WazuhAPIClient:
    """創建 Wazuh 客戶端實例"""
    client = WazuhAPIClient()
    await client.__aenter__()
    return client
