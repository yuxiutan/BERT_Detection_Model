import json
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

# 載入數據（處理JSON Lines格式）
def load_logs(file_path):
    logs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                log = json.loads(line.strip())
                logs.append(log)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line in {file_path}: {e}")
    return logs

# 檔案路徑
four_in_one_logs = load_logs(r"C:\\Liam資料夾\\酪梨工作\\交付文件_v2\\Realtime_Model_Detection_Research\\data\\attack_chain_FourInOne.json")
apt29_logs = load_logs(r"C:\\Liam資料夾\\酪梨工作\\交付文件_v2\\Realtime_Model_Detection_Research\\data\\attack_chain_APT29.json")

# 特徵提取函數
def extract_features(logs):
    sequences = []
    for log in logs:
        desc = log.get("rule.description", "")
        src_ip = log.get("data.srcip", "None")
        dst_ip = log.get("data.dstip", "None")
        sequence = f"{desc} from {src_ip} to {dst_ip}"
        sequences.append(sequence)
    full_sequence = " ".join(sequences)
    return full_sequence

# 提取兩個攻擊鏈的特徵
four_seq = extract_features(four_in_one_logs)
apt_seq = extract_features(apt29_logs)

# 生成合成數據（為了增加樣本數）
def generate_synthetic_sequences(base_seq, label, num_samples=120):  # 增加到120以改善泛化
    data = []
    for i in range(num_samples):
        words = base_seq.split()
        np.random.shuffle(words)
        perturbed = " ".join(words[:int(len(words)*0.9)])  # 截斷10%
        data.append({"sequence": perturbed, "label": label})
    return data

synthetic_four = generate_synthetic_sequences(four_seq, 0)  # 0 for FourInOne
synthetic_apt = generate_synthetic_sequences(apt_seq, 1)     # 1 for APT29

# 合併數據
all_data = synthetic_four + synthetic_apt

# Tokenize序列
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenized_data = [tokenizer(item["sequence"], padding="max_length", truncation=True, max_length=512, return_tensors="pt") for item in all_data]

# 準備標籤
labels = np.array([item["label"] for item in all_data])

# 分割訓練/測試集
train_inputs, test_inputs, train_labels, test_labels = train_test_split(tokenized_data, labels, test_size=0.2, random_state=42)

# 保存預處理數據
with open(r"C:\\Liam資料夾\\酪梨工作\\交付文件_v2\\Realtime_Model_Detection_Research\\Model_v2\\processed_data.pkl", "wb") as f:
    pickle.dump({
        "train_inputs": train_inputs,
        "test_inputs": test_inputs,
        "train_labels": train_labels,
        "test_labels": test_labels
    }, f)

# 生成並保存參考嵌入（使用BERT模型）
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

def get_embedding(sequence):
    inputs = tokenizer(sequence, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # 平均池化
    return embedding

four_embedding = get_embedding(four_seq)
apt_embedding = get_embedding(apt_seq)

with open(r"C:\\Liam資料夾\\酪梨工作\\交付文件_v2\\Realtime_Model_Detection_Research\\Model_v2\\reference_embeddings.pkl", "wb") as f:
    pickle.dump({
        "four_embedding": four_embedding,
        "apt_embedding": apt_embedding
    }, f)

print("Data preprocessing completed. Train size:", len(train_inputs))
print("Reference embeddings saved for similarity calculation.")
