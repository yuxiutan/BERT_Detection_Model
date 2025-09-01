import pickle
import torch
from transformers import BertModel, BertTokenizer
from Model_API.prepare_data import prepare_training_data

def retrain_model(data, base_processed, base_embeddings, output_processed, output_embeddings):
    """重新訓練模型"""
    # 載入原始資料
    with open(base_processed, 'rb') as f:
        processed_data = pickle.load(f)
    with open(base_embeddings, 'rb') as f:
        embeddings = pickle.load(f)

    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # 預處理低信心度資料
    train_data = prepare_training_data(data, tokenizer)

    # 簡單訓練邏輯
    model.train()
    for epoch in range(3):  # 假設 3 個 epoch
        for batch in train_data:
            optimizer.zero_grad()
            outputs = model(**batch['input'])
            loss = compute_loss(outputs, batch['label'], embeddings)  # 需實現
            loss.backward()
            optimizer.step()

    # 更新 embeddings 或 processed_data（根據你的邏輯）
    new_processed_data = processed_data  # 假設更新為新資料
    new_embeddings = embeddings + [outputs.last_hidden_state.mean(dim=1) for batch in train_data]  # 範例：追加新嵌入

    # 儲存新權重
    with open(output_processed, 'wb') as f:
        pickle.dump(new_processed_data, f)
    with open(output_embeddings, 'wb') as f:
        pickle.dump(new_embeddings, f)
    print(f"儲存新權重: {output_processed}, {output_embeddings}")

def compute_loss(outputs, labels, embeddings):
    """計算損失（需根據你的模型實現，例如對比損失）"""
    # 範例：簡單 MSE 損失
    target_embedding = embeddings[0]  # 假設
    return torch.nn.MSELoss()(outputs.last_hidden_state.mean(dim=1), target_embedding)  # 替換實際邏輯
