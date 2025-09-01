#!/bin/bash

# BERT Detection Model 啟動腳本

echo "🚀 正在啟動 BERT Detection Model 服務..."

# 啟動 cron 服務
echo "⏰ 啟動 cron 服務（用於定時重訓練）..."
service cron start

# 檢查環境變數
echo "🔧 檢查環境變數..."
required_vars=("WAZUH_API_URL" "WAZUH_API_USERNAME" "WAZUH_API_PASSWORD")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ 錯誤: 環境變數 $var 未設定"
        exit 1
    fi
    echo "✅ $var 已設定"
done

# 檢查模型檔案
echo "🔍 檢查模型檔案..."
if [ ! -f "/app/Model/bert_model.pth" ] && [ ! -f "/app/Model/bert_model_current.pth" ]; then
    echo "❌ 錯誤: 找不到模型檔案"
    exit 1
fi

# 創建必要的目錄
echo "📁 創建必要的目錄..."
mkdir -p /app/logs
mkdir -p /app/model_backups  
mkdir -p /app/low_confidence_data
mkdir -p /app/data

# 設定權限
echo "🔒 設定檔案權限..."
chmod +x /app/scripts/*.py
chmod +x /app/scripts/*.sh
chmod 666 /app/logs/*

# 測試 Wazuh API 連接
echo "🌐 測試 Wazuh API 連接..."
python3 -c "
import sys
sys.path.insert(0, '/app')
from utils.wazuh_api import WazuhAPIClient
import asyncio

async def test_connection():
    try:
        client = WazuhAPIClient()
        async with client:
            events = await client.get_recent_events(1)
            print(f'✅ Wazuh API 連接成功，獲取到 {len(events)} 個事件')
            return True
    except Exception as e:
        print(f'❌ Wazuh API 連接失敗: {str(e)}')
        return False

result = asyncio.run(test_connection())
sys.exit(0 if result else 1)
"

if [ $? -ne 0 ]; then
    echo "⚠️ 警告: Wazuh API 連接測試失敗，但繼續啟動服務..."
fi

# 設定 Python 路徑
export PYTHONPATH="/app:$PYTHONPATH"

echo "🎯 啟動 FastAPI 服務..."
echo "📊 API 文檔將可在 http://localhost:8000/docs 查看"
echo "🔍 定時檢測間隔: ${DETECTION_INTERVAL:-180} 秒"
echo "📏 信心度閾值: ${CONFIDENCE_THRESHOLD:-0.3}"

# 啟動 FastAPI 應用
exec python3 -m uvicorn main_api:app --host 0.0.0.0 --port 8000 --workers 1
