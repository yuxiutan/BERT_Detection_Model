import json
from transformers import BertTokenizer, BertModel
import torch
from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd
import datetime
import requests
import time
import os

# 設定matplotlib支援中文並關閉顯示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')  # 使用default避免seaborn版本問題
plt.ioff()  # 關閉互動模式，不顯示圖表

# 設定顏色主題
COLORS = {
    'FourInOne': '#ff6b6b',
    'APT29': '#4ecdc4', 
    'Unknown': '#95a5a6',
    'high_conf': '#e74c3c',
    'medium_conf': '#f39c12',
    'low_conf': '#3498db'
}

# =========================
# Step 1: Load log data
# =========================
def load_logs(file_path):
    """載入日誌數據"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Log file not found: {file_path}")
    
    logs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                log = json.loads(line.strip())
                logs.append(log)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line in {file_path}: {e}")
    return logs

# =========================
# Step 2: Feature extraction
# =========================
def extract_features(logs):
    """提取特徵序列"""
    sequences = []
    for log in logs:
        desc = log.get("rule.description", "")
        src_ip = log.get("data.srcip", "None")
        dst_ip = log.get("data.dstip", "None")
        sequence = f"{desc} from {src_ip} to {dst_ip}"
        sequences.append(sequence)
    return sequences

# =========================
# Step 3: Cosine similarity
# =========================
def cosine_similarity(a, b):
    """計算餘弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =========================
# Step 4: Get BERT embedding
# =========================
def get_embedding(sequence, tokenizer, bert_model):
    """獲取BERT嵌入向量"""
    inputs = tokenizer(sequence, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# =========================
# Step 5: Inference function
# =========================
def predict_attack_chain(sequence, four_embedding, apt_embedding, tokenizer, bert_model, threshold=0.5):
    """預測攻擊鏈類型"""
    emb = get_embedding(sequence, tokenizer, bert_model)
    four_sim = cosine_similarity(emb, four_embedding)
    apt_sim = cosine_similarity(emb, apt_embedding)
    max_sim = max(four_sim, apt_sim)
    pred_chain = "FourInOne" if four_sim > apt_sim else "APT29" if max_sim > threshold else "Unknown"
    return pred_chain, {"FourInOne": four_sim, "APT29": apt_sim}, max_sim

# =========================
# Step 6: Visualization functions
# =========================
def create_confidence_distribution(confidences, predictions, save_path="confidence_distribution.png"):
    """創建信心度分佈圖"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 子圖1: 信心度直方圖
    ax1.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
    ax1.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='Threshold: 0.5')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Confidence Score Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子圖2: 按預測類別的信心度箱型圖
    pred_conf_data = defaultdict(list)
    for pred, conf in zip(predictions, confidences):
        pred_conf_data[pred].append(conf)
    
    if len(pred_conf_data) > 0:
        box_data = [pred_conf_data[pred] for pred in pred_conf_data.keys()]
        labels = list(pred_conf_data.keys())
        colors = [COLORS.get(label, '#95a5a6') for label in labels]
        
        bp = ax2.boxplot(box_data, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax2.set_ylabel('Confidence Score')
    ax2.set_title('Confidence by Prediction Class')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_similarity_scatter(four_sims, apt_sims, predictions, save_path="similarity_scatter.png"):
    """創建相似度散點圖"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 為不同預測類別使用不同顏色和形狀
    for pred_class in set(predictions):
        mask = [p == pred_class for p in predictions]
        four_vals = [four_sims[i] for i, m in enumerate(mask) if m]
        apt_vals = [apt_sims[i] for i, m in enumerate(mask) if m]
        
        marker = 'o' if pred_class == 'FourInOne' else 's' if pred_class == 'APT29' else '^'
        color = COLORS.get(pred_class, '#95a5a6')
        
        ax.scatter(four_vals, apt_vals, c=color, marker=marker, 
                  label=f'{pred_class} (n={sum(mask)})', alpha=0.6, s=50)
    
    # 添加對角線
    max_val = max(max(four_sims), max(apt_sims))
    min_val = min(min(four_sims), min(apt_sims))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal Similarity')
    
    ax.set_xlabel('FourInOne Similarity')
    ax.set_ylabel('APT29 Similarity')
    ax.set_title('Attack Chain Similarity Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加統計信息
    stats_text = f'Total samples: {len(predictions)}\nFourInOne wins: {sum(f > a for f, a in zip(four_sims, apt_sims))}\nAPT29 wins: {sum(a > f for f, a in zip(four_sims, apt_sims))}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_prediction_timeline(logs, predictions, confidences, save_path="prediction_timeline.png"):
    """創建預測時間線圖"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 提取時間戳
    timestamps = []
    for i, log in enumerate(logs):
        try:
            ts_str = log.get("timestamp", "")
            if ts_str:
                # 嘗試解析不同格式的時間戳
                for fmt in ["%Y-%m-%dT%H:%M:%S.%f+0000", "%Y-%m-%dT%H:%M:%S+0000", "%Y-%m-%d %H:%M:%S"]:
                    try:
                        ts = datetime.datetime.strptime(ts_str.replace('Z', '+0000'), fmt)
                        break
                    except:
                        continue
                else:
                    ts = datetime.datetime.now() + datetime.timedelta(seconds=i)
            else:
                ts = datetime.datetime.now() + datetime.timedelta(seconds=i)
            timestamps.append(ts)
        except:
            timestamps.append(datetime.datetime.now() + datetime.timedelta(seconds=i))
    
    # 子圖1: 預測類別時間線
    pred_colors = [COLORS.get(pred, '#95a5a6') for pred in predictions]
    ax1.scatter(timestamps, range(len(timestamps)), c=pred_colors, alpha=0.6, s=30)
    ax1.set_ylabel('Event Index')
    ax1.set_title('Attack Chain Predictions Over Time')
    
    # 創建圖例
    legend_elements = [mpatches.Patch(color=COLORS[pred], label=pred) 
                      for pred in set(predictions) if pred in COLORS]
    ax1.legend(handles=legend_elements, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 子圖2: 信心度時間線
    conf_colors = ['#e74c3c' if c >= 0.7 else '#f39c12' if c >= 0.5 else '#3498db' for c in confidences]
    ax2.scatter(timestamps, confidences, c=conf_colors, alpha=0.6, s=30)
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Threshold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Confidence Score')
    ax2.set_title('Confidence Scores Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 格式化x軸
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_high_confidence_analysis(logs, predictions, confidences, threshold=0.45, save_path="high_confidence_analysis.png"):
    """創建高信心度事件分析圖"""
    high_conf_idx = [i for i, conf in enumerate(confidences) if conf >= threshold or predictions[i] != "Unknown"]
    
    if not high_conf_idx:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子圖1: 高信心度事件的源IP分布
    src_ips = [logs[i].get("data.srcip", "Unknown") for i in high_conf_idx]
    src_counter = Counter(src_ips).most_common(10)
    
    if src_counter:
        ips = [ip for ip, _ in src_counter]
        counts = [count for _, count in src_counter]
        ax1.barh(range(len(src_counter)), counts, color='lightcoral', alpha=0.7)
        ax1.set_yticks(range(len(src_counter)))
        ax1.set_yticklabels(ips)
        ax1.set_xlabel('Count')
        ax1.set_title('Top Source IPs in High-Confidence Events')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No source IP data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Top Source IPs in High-Confidence Events')
    
    # 子圖2: 規則ID分布
    rule_ids = [str(logs[i].get("rule.id", "Unknown")) for i in high_conf_idx]
    rule_counter = Counter(rule_ids).most_common(10)
    
    if rule_counter:
        rules = [rule_id for rule_id, _ in rule_counter]
        counts = [count for _, count in rule_counter]
        ax2.bar(range(len(rule_counter)), counts, color='lightblue', alpha=0.7)
        ax2.set_xticks(range(len(rule_counter)))
        ax2.set_xticklabels(rules, rotation=45, ha='right')
        ax2.set_ylabel('Count')
        ax2.set_title('Rule ID Distribution in High-Confidence Events')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No rule ID data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Rule ID Distribution in High-Confidence Events')
    
    # 子圖3: 信心度 vs 規則等級
    rule_levels = [logs[i].get("rule.level", 0) for i in high_conf_idx]
    high_confidences = [confidences[i] for i in high_conf_idx]
    high_predictions = [predictions[i] for i in high_conf_idx]
    
    colors = [COLORS.get(pred, '#95a5a6') for pred in high_predictions]
    ax3.scatter(rule_levels, high_confidences, c=colors, alpha=0.6, s=50)
    ax3.set_xlabel('Rule Level')
    ax3.set_ylabel('Confidence Score')
    ax3.set_title('Confidence vs Rule Level')
    ax3.grid(True, alpha=0.3)
    
    # 子圖4: 攻擊鏈類型分布餅圖
    pred_counter = Counter([predictions[i] for i in high_conf_idx])
    if pred_counter:
        colors_pie = [COLORS.get(pred, '#95a5a6') for pred in pred_counter.keys()]
        wedges, texts, autotexts = ax4.pie(pred_counter.values(), labels=pred_counter.keys(), 
                                          colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Attack Chain Distribution\n(High-Confidence Events)')
    else:
        ax4.text(0.5, 0.5, 'No predictions to display', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Attack Chain Distribution\n(High-Confidence Events)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 關閉圖表，釋放記憶體
    # print(f"📊 High-confidence analysis chart saved: {save_path}")  # 移除print輸出

def create_model_performance_summary(predictions, confidences, four_sims, apt_sims, save_path="model_performance.png"):
    """創建模型性能總結圖"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # 子圖1: 預測類別分布
    pred_counts = Counter(predictions)
    colors = [COLORS.get(pred, '#95a5a6') for pred in pred_counts.keys()]
    wedges, texts, autotexts = ax1.pie(pred_counts.values(), labels=pred_counts.keys(), 
                                      colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Overall Prediction Distribution')
    
    # 子圖2: 信心度等級分布
    conf_levels = ['High (≥0.7)' if c >= 0.7 else 'Medium (0.5-0.7)' if c >= 0.5 else 'Low (<0.5)' 
                   for c in confidences]
    conf_counts = Counter(conf_levels)
    level_colors = ['#e74c3c', '#f39c12', '#3498db']
    
    # 確保所有等級都有對應的顏色
    ordered_levels = ['High (≥0.7)', 'Medium (0.5-0.7)', 'Low (<0.5)']
    ordered_counts = [conf_counts.get(level, 0) for level in ordered_levels]
    
    ax2.bar(ordered_levels, ordered_counts, color=level_colors, alpha=0.7)
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Level Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 子圖3: 相似度統計
    sim_stats = {
        'FourInOne': {
            'Mean': np.mean(four_sims),
            'Std': np.std(four_sims),
            'Max': np.max(four_sims),
            'Min': np.min(four_sims)
        },
        'APT29': {
            'Mean': np.mean(apt_sims),
            'Std': np.std(apt_sims), 
            'Max': np.max(apt_sims),
            'Min': np.min(apt_sims)
        }
    }
    
    metrics = ['Mean', 'Std', 'Max', 'Min']
    four_vals = [sim_stats['FourInOne'][m] for m in metrics]
    apt_vals = [sim_stats['APT29'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax3.bar(x - width/2, four_vals, width, label='FourInOne', color=COLORS['FourInOne'], alpha=0.7)
    ax3.bar(x + width/2, apt_vals, width, label='APT29', color=COLORS['APT29'], alpha=0.7)
    
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Similarity Score')
    ax3.set_title('Similarity Statistics Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子圖4: 性能指標文字摘要
    ax4.axis('off')
    
    # 計算關鍵指標
    total_samples = len(predictions)
    high_conf_count = sum(1 for c in confidences if c >= 0.7)
    unknown_rate = predictions.count('Unknown') / total_samples * 100
    avg_confidence = np.mean(confidences)
    
    summary_text = f"""MODEL PERFORMANCE SUMMARY

Total Samples: {total_samples:,}
High Confidence (≥0.7): {high_conf_count} ({high_conf_count/total_samples*100:.1f}%)
Unknown Predictions: {predictions.count('Unknown')} ({unknown_rate:.1f}%)
Average Confidence: {avg_confidence:.3f}

Dominant Attack Type: {max(pred_counts, key=pred_counts.get) if pred_counts else 'None'}
FourInOne vs APT29: {predictions.count('FourInOne')}:{predictions.count('APT29')}

Similarity Statistics:
• FourInOne Mean: {np.mean(four_sims):.3f}
• APT29 Mean: {np.mean(apt_sims):.3f}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
            verticalalignment='top', fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def interactive_threshold_analysis(confidences, predictions, four_sims, apt_sims, save_path="threshold_analysis.png"):
    """互動式閾值分析 - 幫助找到最佳閾值"""
    thresholds = np.arange(0.1, 1.0, 0.05)
    results = []
    
    for threshold in thresholds:
        high_conf_predictions = [pred if conf >= threshold else "Unknown" 
                                for pred, conf in zip(predictions, confidences)]
        
        known_count = sum(1 for pred in high_conf_predictions if pred != "Unknown")
        four_count = high_conf_predictions.count("FourInOne")
        apt_count = high_conf_predictions.count("APT29")
        
        results.append({
            'threshold': threshold,
            'known_count': known_count,
            'four_count': four_count,
            'apt_count': apt_count,
            'known_rate': known_count / len(predictions) if len(predictions) > 0 else 0
        })
    
    # 繪製閾值分析圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    thresholds_list = [r['threshold'] for r in results]
    known_rates = [r['known_rate'] for r in results]
    known_counts = [r['known_count'] for r in results]
    
    ax1.plot(thresholds_list, known_rates, 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Known Prediction Rate')
    ax1.set_title('Prediction Rate vs Threshold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='50% Rate')
    ax1.legend()
    
    ax2.plot(thresholds_list, known_counts, 'g-s', linewidth=2, markersize=4)
    ax2.set_xlabel('Confidence Threshold')
    ax2.set_ylabel('Number of Known Predictions')
    ax2.set_title('Prediction Count vs Threshold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if results:
        optimal_idx = np.argmax([r['known_rate'] * r['known_count'] for r in results])
        optimal_threshold = results[optimal_idx]['threshold']
        
        return optimal_threshold
    else:
        return 0.5

def generate_analysis_report(logs, predictions, confidences, four_sims, apt_sims, output_dir="analysis_charts"):
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        create_confidence_distribution(confidences, predictions, 
                                      save_path=os.path.join(output_dir, "01_confidence_distribution.png"))
        
        create_similarity_scatter(four_sims, apt_sims, predictions,
                                 save_path=os.path.join(output_dir, "02_similarity_scatter.png"))
        
        create_prediction_timeline(logs, predictions, confidences,
                                  save_path=os.path.join(output_dir, "03_prediction_timeline.png"))
        
        create_high_confidence_analysis(logs, predictions, confidences,
                                       save_path=os.path.join(output_dir, "04_high_confidence_analysis.png"))
        
        create_model_performance_summary(predictions, confidences, four_sims, apt_sims,
                                        save_path=os.path.join(output_dir, "05_model_performance.png"))
        
    except Exception as e:
        import traceback
        pass

# =========================
# Main execution
# =========================
def main():
    try:
        # 設定檔案路徑（可修改為你的實際路徑）
        log_file = r"C:\\Liam資料夾\\酪梨工作\\交付文件_v2\\Realtime_Model_Detection_Research\\data\\attack_chain_FourInOne.json"
        ref_path = Path(r"C:\\Liam資料夾\\酪梨工作\\交付文件_v2\\Realtime_Model_Detection_Research\\Model_v2\\reference_embeddings.pkl")
        
        # 檢查檔案是否存在
        if not os.path.exists(log_file):
            print(f"Log file not found: {log_file}")
            return
            
        if not ref_path.exists():
            print(f"Reference embeddings file not found: {ref_path}")
            return

        # Load logs
        logs = load_logs(log_file)
        sequences = extract_features(logs)

        if len(logs) == 0:
            print("No logs found in file")
            return

        # Load BERT
        try:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            bert_model = BertModel.from_pretrained("bert-base-uncased")
            bert_model.eval()
        except Exception as e:
            print(f"Failed to load BERT model: {e}")
            return

        # Load reference embeddings
        with open(ref_path, "rb") as f:
            ref_data = pickle.load(f)
            four_emb = ref_data["four_embedding"]
            apt_emb = ref_data["apt_embedding"]

        # =========================
        # Step 6: Run inference
        # =========================
        predictions, four_sims, apt_sims, confidences = [], [], [], []
        
        for i, seq in enumerate(sequences):
            pred, sims, conf = predict_attack_chain(seq, four_emb, apt_emb, tokenizer, bert_model)
            predictions.append(pred)
            four_sims.append(sims["FourInOne"])
            apt_sims.append(sims["APT29"])
            confidences.append(conf)

        # 修正：使用正確的變量名
        max_idx = np.argmax(confidences)
        final_pred = predictions[max_idx]
        final_confidence = confidences[max_idx]
        
        # 計算平均信心度
        known_confidences = [confidences[i] for i, p in enumerate(predictions) if p != "Unknown"]
        avg_conf = np.mean(known_confidences) if known_confidences else 0
        
        # Output summary - 只保留重要信息
        print(f"Predicted Attack Chain: {final_pred}")
        print(f"Max Confidence: {final_confidence:.4f}")
        print(f"Average Confidence: {avg_conf:.4f}")
        
        if avg_conf > 0.5:
            print("ALERT: High confidence detection!")
        else:
            print("No high-confidence alert generated")

        # =========================
        # Step 7: Generate Analysis Charts (靜默執行)
        # =========================
        generate_analysis_report(logs, predictions, confidences, four_sims, apt_sims)
        
        # =========================
        # Step 7.5: Interactive Threshold Analysis (靜默執行)
        # =========================
        optimal_threshold = interactive_threshold_analysis(confidences, predictions, four_sims, apt_sims,
                                                          save_path="analysis_charts/06_threshold_analysis.png")

        # =========================
        # Step 8: Prepare time format
        # =========================
        now = datetime.datetime.now(datetime.timezone.utc)
        formatted_time = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000"

        # =========================
        # Step 9: Collect high-confidence events
        # =========================
        high_idx = [i for i, conf in enumerate(confidences) if conf >= 0.45 or predictions[i] != "Unknown"]
        alerts = []
        
        for idx in high_idx:
            log = logs[idx]
            individual_alert = {
                "timestamp": log.get("timestamp", formatted_time),
                "rule_id": log.get("rule.id", ""),
                "rule_level": log.get("rule.level", ""),
                "description": log.get("rule.description", ""),
                "src_ip": log.get("data.srcip", "None"),
                "dst_ip": log.get("data.dstip", "None"),
                "full_log": log.get("full_log", "")
            }
            alerts.append(individual_alert)

        # =========================
        # Step 10: Prepare correlation alert
        # =========================
        if high_idx:  # 確保有高信心度事件
            max_log = logs[max_idx]
            involved_info = []

            for idx in high_idx:
                log = logs[idx]
                agent_id = log.get("agent.id", "Unknown")
                src_ip = log.get("data.srcip", "None")
                dst_ip = log.get("data.dstip", "None")
                conf = confidences[idx]
                pred = predictions[idx]
                involved_info.append(f"Agent ID: {agent_id}, Src IP: {src_ip}, Dst IP: {dst_ip}, Predicted: {pred}, Confidence: {conf:.4f}")

            full_log_str = f"The model detected {final_pred} attack chain. Details: {'; '.join(involved_info)}. Average confidence: {avg_conf:.4f}"
            correlation_description = f"The model detected {final_pred} attack chain"

            correlation_alert = {
                "timestamp": formatted_time,
                "rule_id": 51110,
                "rule_level": 12,
                "description": correlation_description,
                "src_ip": max_log.get("data.srcip", "None"),
                "dst_ip": max_log.get("data.dstip", "None"),
                "full_log": full_log_str
            }

            # =========================
            # Step 11: Organize data
            # =========================
            data = {
                "data": [{
                    "correlation_alert": correlation_alert,
                    "alerts": alerts
                }]
            }

            # =========================
            # Step 12: Save JSON file
            # =========================
            output_path = "high_confidence_logs.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            # =========================
            # Step 13: POST to API (靜默執行，只在錯誤時輸出)
            # =========================
            post_url = "http://100.89.12.61:8999/newalert"
            headers = {"Content-Type": "application/json"}
            
            try:
                resp = requests.post(post_url, headers=headers, json=data, timeout=10)
                if resp.status_code != 200:
                    print(f"API POST failed with status {resp.status_code}")
                    
            except requests.exceptions.Timeout:
                print("API request timed out")
            except requests.exceptions.ConnectionError:
                print("API connection error")
            except Exception as e:
                print(f"API POST failed: {e}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
