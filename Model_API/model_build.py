import os
import pickle
import glob
from transformers import BertModel, BertTokenizer  # 假設使用 BERT

def load_model():
    """載入最新模型和權重"""
    weights_dir = os.environ['MODEL_WEIGHTS_DIR']
    orig_processed = os.environ['ORIGINAL_PROCESSED_DATA']
    orig_embeddings = os.environ['ORIGINAL_REFERENCE_EMBEDDINGS']

    # 尋找最新權重
    processed_files = sorted(glob.glob(f"{weights_dir}/processed_data_v*.pkl"), key=os.path.getmtime)
    embeddings_files = sorted(glob.glob(f"{weights_dir}/reference_embeddings_v*.pkl"), key=os.path.getmtime)

    processed_path = processed_files[-1] if processed_files else orig_processed
    embeddings_path = embeddings_files[-1] if embeddings_files else orig_embeddings

    with open(processed_path, 'rb') as f:
        processed_data = pickle.load(f)
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)

    # 假設：初始化 BERT 模型
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model.eval()
    return {
        'model': model,
        'tokenizer': tokenizer,
        'processed_data': processed_data,
        'embeddings': embeddings
    }
