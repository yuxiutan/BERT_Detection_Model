#!/bin/bash

# Wazuh API 連線資訊
WAZUH_USER="admin"
WAZUH_PASSWORD="SecretPassword"
WAZUH_API_URL="https://100.79.144.59:9200/wazuh-alerts-*/_search"

# 輸出檔案
OUTPUT_FILE="/app/data/new_attack_data.jsonl"
: > "$OUTPUT_FILE"  # 清空舊檔案

# search_after 初始值
SEARCH_AFTER=""

TOTAL=0

while true; do
    # 建立查詢 JSON
    read -r -d '' QUERY_DATA <<EOF
{
  "query": {
    "bool": {
      "must": [
        {
          "range": {
            "timestamp": {
              "gte": "now-5m",
              "lte": "now"
            }
          }
        },
        {
          "terms": {
            "agent.name": [
              "DESKTOP-66TG6SE",
              "DESKTOP-66G2GGG",
              "connector-node"
            ]
          }
        }
      ]
    }
  },
  "_source": [
    "timestamp",
    "agent.ip",
    "agent.name",
    "agent.id",
    "rule.id",
    "rule.mitre.id",
    "rule.level",
    "rule.description",
    "data.srcip",
    "data.dstip",
    "full_log"
  ],
  "sort": [
    { "timestamp": { "order": "asc" } }
  ],
  "size": 10000
  $(if [ -n "$SEARCH_AFTER" ]; then echo ", \"search_after\": $SEARCH_AFTER"; fi)
}
EOF

    # 呼叫 API
    RESPONSE=$(curl -s -k -u "$WAZUH_USER:$WAZUH_PASSWORD" -X POST "$WAZUH_API_URL" \
      -H 'Content-Type: application/json' \
      -d "$QUERY_DATA")

    # 取出 hits
    HITS=$(echo "$RESPONSE" | jq -c '.hits.hits')

    # 如果沒有資料就結束
    if [ "$(echo "$HITS" | jq 'length')" -eq 0 ]; then
        echo "抓取完成，總共 $TOTAL 筆資料。"
        break
    fi

    # 寫入檔案
    echo "$HITS" | jq -c '.[] | ._source | {
        "timestamp": .timestamp,
        "agent.ip": .agent.ip,
        "agent.name": .agent.name,
        "agent.id": .agent.id,
        "rule.id": .rule.id,
        "rule.mitre.id": (if .rule.mitre.id == null then "T0000" else .rule.mitre.id end),
        "rule.level": .rule.level,
        "rule.description": .rule.description,
        "data.srcip": .data.srcip,
        "data.dstip": .data.dstip,
        "full_log": .full_log
    }' >> "$OUTPUT_FILE"

    # 更新總數
    COUNT=$(echo "$HITS" | jq 'length')
    TOTAL=$((TOTAL + COUNT))
    echo "已抓取 $COUNT 筆，累計 $TOTAL 筆..."

    # 更新 search_after（取最後一筆 sort 值）
    SEARCH_AFTER=$(echo "$HITS" | jq -c '.[-1].sort')

    # 如果 search_after 為 null，結束
    if [ "$SEARCH_AFTER" = "null" ] || [ -z "$SEARCH_AFTER" ]; then
        echo "沒有更多資料了。"
        break
    fi
done
