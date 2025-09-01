#!/usr/bin/env python3
"""
重訓練排程器
每天晚上21:00自動執行重訓練任務
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timezone

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, '/app')

from scripts.retrain_manager import RetrainManager

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/retrain_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """主要的重訓練排程任務"""
    try:
        logger.info("🕘 重訓練排程器開始執行...")
        logger.info(f"⏰ 執行時間: {datetime.now(timezone.utc).isoformat()}")
        
        # 初始化重訓練管理器
        retrain_manager = RetrainManager()
        
        # 執行重訓練
        success = await retrain_manager.retrain_model()
        
        if success:
            logger.info("✅ 排程重訓練任務完成")
            print("SUCCESS: Retrain completed successfully")
        else:
            logger.error("❌ 排程重訓練任務失敗")
            print("ERROR: Retrain failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ 重訓練排程器異常: {str(e)}")
        print(f"ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # 運行異步主函數
    asyncio.run(main())
