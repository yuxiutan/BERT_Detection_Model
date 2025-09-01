import torch
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import pickle
import numpy as np
import json
from tqdm import tqdm
import os
from pathlib import Path
from playwright.sync_api import sync_playwright, Playwright

# 定義保存圖表的目錄
PLOTS_DIR = Path(r"C:\Liam資料夾\酪梨工作\交付文件_v2\Realtime_Model_Detection_Research\Model_v2\plots")
PLOTS_DIR.mkdir(exist_ok=True)  # 創建目錄（如果不存在）

# HTML 模板，包含 Chart.js
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <canvas id="chart" width="800" height="600"></canvas>
    <script>
        const ctx = document.getElementById('chart').getContext('2d');
        new Chart(ctx, {chart_config});
    </script>
</body>
</html>"""

# 函數：將 Chart.js 配置渲染為 PNG
def save_chart_as_png(chart_config, title, output_path):
    try:
        # 臨時保存 HTML
        temp_html_path = PLOTS_DIR / f"temp_{title.replace(' ', '_').lower()}.html"
        with open(temp_html_path, "w", encoding="utf-8") as f:
            f.write(HTML_TEMPLATE.format(title=title, chart_config=json.dumps(chart_config)))
        
        # 使用 Playwright 渲染並截圖
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"file://{temp_html_path.resolve()}")
            page.wait_for_timeout(2000)  # 增加等待時間確保渲染
            page.screenshot(path=output_path, full_page=True)
            browser.close()
        
        # 刪除臨時 HTML
        os.remove(temp_html_path)
        print(f"Saved chart as PNG: {output_path}")
    except Exception as e:
        print(f"Failed to save chart {title} as PNG: {e}")
        # 保存為 HTML 作為備案
        html_path = PLOTS_DIR / f"{title.replace(' ', '_').lower()}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(HTML_TEMPLATE.format(title=title, chart_config=json.dumps(chart_config)))
        print(f"Saved chart as HTML: {html_path}")

try:
    # 載入預處理數據
    with open(r"C:\Liam資料夾\酪梨工作\交付文件_v2\Realtime_Model_Detection_Research\Model_v2\processed_data.pkl", "rb") as f:
        data = pickle.load(f)
        train_inputs = data["train_inputs"]
        test_inputs = data["test_inputs"]
        train_labels = data["train_labels"]
        test_labels = data["test_labels"]

    # 轉換成TensorDataset
    input_ids = torch.cat([item['input_ids'] for item in train_inputs], dim=0)
    attention_masks = torch.cat([item['attention_mask'] for item in train_inputs], dim=0)
    train_labels_tensor = torch.tensor(train_labels)

    dataset = TensorDataset(input_ids, attention_masks, train_labels_tensor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 載入模型
    print("Using device: cpu")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, hidden_dropout_prob=0.3)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # 訓練迴圈並記錄損失
    model.train()
    train_losses = []
    for epoch in range(3):  # 3個epoch
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            b_input_ids, b_attention_mask, b_labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({"batch_loss": loss.item()})
        avg_loss = total_loss / len(dataloader)
        train_losses.append(float(avg_loss))  # 轉為 Python float
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # 評估（在測試集上）
    test_input_ids = torch.cat([item['input_ids'] for item in test_inputs], dim=0)
    test_attention_masks = torch.cat([item['attention_mask'] for item in test_inputs], dim=0)
    test_labels_tensor = torch.tensor(test_labels)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=test_input_ids, attention_mask=test_attention_masks)
        probs = torch.softmax(outputs.logits, dim=1).numpy()  # 機率分數
        preds = torch.argmax(outputs.logits, dim=1).numpy()  # 預測標籤

    acc = accuracy_score(test_labels, preds)
    print("Test Accuracy:", acc)

    # 保存模型
    model.save_pretrained(r"C:\Liam資料夾\酪梨工作\交付文件_v2\Realtime_Model_Detection_Research\Model_v2\attack_chain_classifier")

    # 1. 混淆矩陣
    cm = confusion_matrix(test_labels, preds)
    cm_data = cm.tolist()  # 轉為 Python list
    cm_config = {
        "type": "bar",
        "data": {
            "labels": ["FourInOne", "APT29"],
            "datasets": [
                {
                    "label": "Predicted FourInOne",
                    "data": [int(cm_data[0][0]), int(cm_data[1][0])],  # 轉為 int
                    "backgroundColor": "#1E90FF"
                },
                {
                    "label": "Predicted APT29",
                    "data": [int(cm_data[0][1]), int(cm_data[1][1])],  # 轉為 int
                    "backgroundColor": "#FF4500"
                }
            ]
        },
        "options": {
            "indexAxis": "y",
            "scales": {
                "x": {"stacked": True, "title": {"display": True, "text": "Count"}},
                "y": {"stacked": True, "title": {"display": True, "text": "True Label"}}
            },
            "plugins": {
                "title": {"display": True, "text": "Confusion Matrix"},
                "legend": {"position": "top"}
            }
        }
    }
    save_chart_as_png(cm_config, "Confusion Matrix", PLOTS_DIR / "confusion_matrix.png")

    # 2. ROC 曲線
    fpr, tpr, _ = roc_curve(test_labels, probs[:, 1])
    roc_auc = float(auc(fpr, tpr))  # 轉為 float
    roc_config = {
        "type": "line",
        "data": {
            "labels": [float(x) for x in fpr],  # 轉為 Python list of floats
            "datasets": [
                {
                    "label": f"APT29 (AUC = {roc_auc:.3f})",
                    "data": [float(x) for x in tpr],  # 轉為 Python list of floats
                    "borderColor": "#1E90FF",
                    "fill": False
                },
                {
                    "label": "Random",
                    "data": [float(x) for x in fpr],  # 對角線
                    "borderColor": "#000000",
                    "borderDash": [5, 5],
                    "fill": False
                }
            ]
        },
        "options": {
            "scales": {
                "x": {"title": {"display": True, "text": "False Positive Rate"}},
                "y": {"title": {"display": True, "text": "True Positive Rate"}, "max": 1.05}
            },
            "plugins": {
                "title": {"display": True, "text": "ROC Curve"},
                "legend": {"position": "top"}
            }
        }
    }
    save_chart_as_png(roc_config, "ROC Curve", PLOTS_DIR / "roc_curve.png")

    # 3. Precision-Recall 曲線
    precision, recall, _ = precision_recall_curve(test_labels, probs[:, 1])
    pr_auc = float(auc(recall, precision))  # 轉為 float
    pr_config = {
        "type": "line",
        "data": {
            "labels": [float(x) for x in recall],  # 轉為 Python list of floats
            "datasets": [
                {
                    "label": f"APT29 (AUC = {pr_auc:.3f})",
                    "data": [float(x) for x in precision],  # 轉為 Python list of floats
                    "borderColor": "#1E90FF",
                    "fill": False
                }
            ]
        },
        "options": {
            "scales": {
                "x": {"title": {"display": True, "text": "Recall"}},
                "y": {"title": {"display": True, "text": "Precision"}}
            },
            "plugins": {
                "title": {"display": True, "text": "Precision-Recall Curve"},
                "legend": {"position": "top"}
            }
        }
    }
    save_chart_as_png(pr_config, "Precision-Recall Curve", PLOTS_DIR / "precision_recall_curve.png")

    # 4. 訓練歷史（損失）
    loss_config = {
        "type": "line",
        "data": {
            "labels": [1, 2, 3],
            "datasets": [
                {
                    "label": "Training Loss",
                    "data": train_losses,  # 已轉為 float
                    "borderColor": "#1E90FF",
                    "fill": False
                }
            ]
        },
        "options": {
            "scales": {
                "x": {"title": {"display": True, "text": "Epoch"}},
                "y": {"title": {"display": True, "text": "Loss"}}
            },
            "plugins": {
                "title": {"display": True, "text": "Training Loss History"},
                "legend": {"position": "top"}
            }
        }
    }
    save_chart_as_png(loss_config, "Training Loss History", PLOTS_DIR / "training_loss_history.png")

    print(f"All charts saved as PNG to {PLOTS_DIR}")

except Exception as e:
    print(f"Training failed: {e}")
