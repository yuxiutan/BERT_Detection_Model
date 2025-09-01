from transformers import BertTokenizer

def preprocess_event(full_log, tokenizer):
    """預處理單一事件為 BERT 輸入"""
    inputs = tokenizer(full_log, return_tensors='pt', max_length=512, truncation=True, padding=True)
    return inputs

def prepare_training_data(data, tokenizer):
    """準備訓練資料"""
    train_data = []
    for event in data:
        inputs = preprocess_event(event['full_log'], tokenizer)
        # 假設標籤（需根據你的模型定義，例如 0 為異常）
        train_data.append({'input': inputs, 'label': 0})  # 替換實際標籤邏輯
    return train_data
