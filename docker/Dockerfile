FROM python:3.9-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    curl \
    cron \
    jq \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt 並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案檔案
COPY . .

# 創建必要的目錄
RUN mkdir -p /app/logs \
    /app/model_backups \
    /app/low_confidence_data \
    /app/data

# 設定權限
RUN chmod +x /app/scripts/*.sh

# 創建 cron job 檔案
RUN echo "0 21 * * * /usr/local/bin/python /app/scripts/retrain_scheduler.py >> /app/logs/retrain.log 2>&1" > /etc/cron.d/retrain-job
RUN chmod 0644 /etc/cron.d/retrain-job
RUN crontab /etc/cron.d/retrain-job

# 暴露端口
EXPOSE 8000

# 啟動腳本
CMD ["bash", "/app/scripts/start.sh"]
