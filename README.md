# BERT_Detection_Model

## Project Structure

```bash
Realtime_Transformer_Chain_Detection/
├── Model/                        # Model evaluation outputs
│ ├── processed_data.pkl
│ ├── reference_embeddings.pkl
│── Model_API/
│   ├── app.py                   
│   ├── inference.py
│   ├── model_build.py
│   ├── prepare_data.py
│   └── retrain_model.py
├── data/
│ ├── attack_chain_APT29.json
│ ├── attack_chain_FourInOne.json
│ └── new_attack_data.jsonl    # New log data for evaluation
├── docker/
│ ├── .dockerignore
│ ├── Dockerfile
│ └── docker-compose.yml
├── utils/
│ ├── .env.local
│ ├── clear_log.sh
│ └── wazuh_api.py                  # Wazuh API integration script
├── app.py                          # Dash-based dashboard
└── requirements.txt                # Python dependency list
```
