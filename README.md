# BERT_Detection_Model

Realtime_Transformer_Chain_Detection/
├── Model/                        # Model evaluation outputs
│ ├── Report/
│ │ ├── prediction_confidence_distribution_Chain0.png
│ │ ├── prediction_confidence_distribution_Chain1.png
│ │ ├── prediction_confidence_distribution_Chain2.png
│ │ ├── prediction_results_Chain0.csv
│ │ ├── prediction_results_Chain1.csv
│ │ └── prediction_results_Chain2.csv
│ ├── improved_preprocessors.pkl    # Encoders and transformers
│ ├── improved_transformer_model.keras  # # Trained Transformer model
│ ├── inference.py    # Real-time prediction & scoring
│ ├── model_build.py  # Train Transformer model, output metrics
│── api/
│   ├── inference.py                   # 模型推論
│   ├── transformer_retrain_model.py   # retrain 機制
│   ├── auto_label.py                  # pseudo-label + K-means
│   ├── scheduler.py                   # 自動 retrain 排程
│   └── app.py                         # FastAPI 主服務
├── data/
│ ├── attack_chain_0.jsonl     # Training logs for Chain 0
│ ├── attack_chain_1.jsonl     # Training logs for Chain 1
│ ├── attack_chain_2.jsonl     # Training logs for Chain 2
│ └── new_attack_data.jsonl    # New log data for evaluation
│   ├── auto_labeled_logs.json   # 自動標記 log
│   └── uncertain_logs.json      # 低信心 log
├── docker/
│ ├── .dockerignore
│ ├── Dockerfile
│ ├── docker-compose.yml
│ └──supervisord.conf
├── utils/
│ ├── .env.local
│ ├── clear_log.sh
│ └── wazuh_api.py                  # Wazuh API integration script
├── Model_confusion_matrix.png      # Confusion matrix plot
├── Model_roc_auc.png               # ROC curve and AUC plot
├── Model_training_history.png      # Training loss/accuracy
├── app.py                          # Dash-based dashboard
└── requirements.txt                # Python dependency list
