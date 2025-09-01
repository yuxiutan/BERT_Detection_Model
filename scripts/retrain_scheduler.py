import os
import json
import glob
import shutil
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from utils.wazuh_api import fetch_wazuh_data
from Model_API.inference import run_inference
from Model_API.retrain_model import retrain_model
from Model_API.model_build import load_model

LOW_CONF_FILE = os.environ['LOW_CONFIDENCE_FILE']
WEIGHTS_DIR = os.environ['MODEL_WEIGHTS_DIR']
ORIGINAL_PROCESSED_DATA = os.environ['ORIGINAL_PROCESSED_DATA']
ORIGINAL_REFERENCE_EMBEDDINGS = os.environ['ORIGINAL_REFERENCE_EMBEDDINGS']

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(f"{WEIGHTS_DIR}/backups", exist_ok=True)

def monitor_task():
    """每 3 分鐘獲取資料並推論，低信心度事件存入檔案"""
    events = fetch_wazuh_data()
    if not events:
        return

    model = load_model()  # 載入最新模型
    low_conf_events = []

    for event in events:
        confidence = run_inference(model, event)  # 假設返回信心度
        if confidence < 0.3:
            low_conf_events.append(event)

    if low_conf_events:
        with open(LOW_CONF_FILE, 'a', encoding='utf-8') as f:
            for event in low_conf_events:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        print(f"儲存 {len(low_conf_events)} 筆低信心度事件到 {LOW_CONF_FILE}")

def retrain_task():
    """每天 21:00 使用低信心度事件重新訓練"""
    if not os.path.exists(LOW_CONF_FILE):
        print("無低信心度事件，跳過 Retrain")
        return

    with open(LOW_CONF_FILE, 'r', encoding='utf-8') as f:
        low_conf_data = [json.loads(line) for line in f if line.strip()]

    if not low_conf_data:
        print("低信心度事件檔案為空，跳過 Retrain")
        return

    # 備份最新三版權重
    current_processed = sorted(glob.glob(f"{WEIGHTS_DIR}/processed_data_v*.pkl"), key=os.path.getmtime)
    current_embeddings = sorted(glob.glob(f"{WEIGHTS_DIR}/reference_embeddings_v*.pkl"), key=os.path.getmtime)
    
    if len(current_processed) >= 3:
        backups = current_processed[-3:]  # 最新三版
        for i, weight in enumerate(backups):
            shutil.copy(weight, f"{WEIGHTS_DIR}/backups/processed_data_backup_v{i+1}.pkl")
            embedding = weight.replace("processed_data", "reference_embeddings")
            if os.path.exists(embedding):
                shutil.copy(embedding, f"{WEIGHTS_DIR}/backups/reference_embeddings_backup_v{i+1}.pkl")
            print(f"備份 {weight} 和對應嵌入檔案")

    # 重新訓練
    new_version = len(current_processed) + 1
    new_processed_path = f"{WEIGHTS_DIR}/processed_data_v{new_version}.pkl"
    new_embeddings_path = f"{WEIGHTS_DIR}/reference_embeddings_v{new_version}.pkl"
    
    retrain_model(
        low_conf_data,
        base_processed=ORIGINAL_PROCESSED_DATA,
        base_embeddings=ORIGINAL_REFERENCE_EMBEDDINGS,
        output_processed=new_processed_path,
        output_embeddings=new_embeddings_path
    )
    print(f"生成新權重: {new_processed_path}, {new_embeddings_path}")

    # 清空低信心度事件檔案
    open(LOW_CONF_FILE, 'w').close()
    print("清空低信心度事件檔案")

def start_scheduler():
    """啟動排程器"""
    scheduler = BackgroundScheduler()
    scheduler.add_job(monitor_task, 'interval', minutes=3)
    scheduler.add_job(retrain_task, CronTrigger(hour=21, minute=0))
    scheduler.start()
    print("📅 排程器啟動")
