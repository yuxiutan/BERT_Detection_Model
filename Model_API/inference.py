import torch
from Model_API.prepare_data import preprocess_event

def run_inference(model_dict, event):
    """對單一事件進行推論"""
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    embeddings = model_dict['embeddings']
    
    # 預處理事件（假設使用 full_log）
    input_data = preprocess_event(event['full_log'], tokenizer)
    
    with torch.no_grad():
        outputs = model(**input_data)
        # 假設：比較事件嵌入與 reference_embeddings，計算信心度
        event_embedding = outputs.last_hidden_state.mean(dim=1)  # 簡化
        confidence = compute_confidence(event_embedding, embeddings)  # 需實現
    return confidence

def compute_confidence(event_embedding, reference_embeddings):
    """計算信心度（需根據你的邏輯實現，例如餘弦相似度）"""
    # 範例：計算與參考嵌入的平均相似度
    similarities = [torch.cosine_similarity(event_embedding, ref, dim=1).item() for ref in reference_embeddings]
    return max(similarities) if similarities else 0.0  # 替換為實際邏輯，範圍 0-1
