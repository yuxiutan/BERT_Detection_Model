import os
import requests
import json

def fetch_wazuh_data():
    """從 Wazuh API 獲取最近 5 分鐘的事件資料"""
    url = os.environ['WAZUH_API_URL']
    user = os.environ['WAZUH_USER']
    password = os.environ['WAZUH_PASSWORD']
    output_file = os.environ['OUTPUT_FILE']

    query_data = {
        "query": {
            "bool": {
                "must": [
                    {"range": {"timestamp": {"gte": "now-5m", "lte": "now"}}},
                    {"terms": {"agent.name": ["DESKTOP-66TG6SE", "DESKTOP-66G2GGG", "connector-node"]}}
                ]
            }
        },
        "_source": [
            "timestamp", "agent.ip", "agent.name", "agent.id",
            "rule.id", "rule.mitre.id", "rule.level", "rule.description",
            "data.srcip", "data.dstip", "full_log"
        ],
        "sort": [{"timestamp": {"order": "asc"}}],
        "size": 10000
    }

    total = 0
    search_after = None
    data_list = []

    while True:
        if search_after:
            query_data['search_after'] = search_after

        try:
            response = requests.post(url, auth=(user, password), json=query_data, verify=False)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"❌ API 請求失敗: {e}")
            return []

        hits = response.json().get('hits', {}).get('hits', [])
        if not hits:
            break

        for hit in hits:
            source = hit['_source']
            event = {
                "timestamp": source.get('timestamp'),
                "agent.ip": source.get('agent', {}).get('ip'),
                "agent.name": source.get('agent', {}).get('name'),
                "agent.id": source.get('agent', {}).get('id'),
                "rule.id": source.get('rule', {}).get('id'),
                "rule.mitre.id": source.get('rule', {}).get('mitre', {}).get('id', 'T0000'),
                "rule.level": source.get('rule', {}).get('level'),
                "rule.description": source.get('rule', {}).get('description'),
                "data.srcip": source.get('data', {}).get('srcip'),
                "data.dstip": source.get('data', {}).get('dstip'),
                "full_log": source.get('full_log')
            }
            data_list.append(event)

        total += len(hits)
        search_after = hits[-1]['sort'] if hits else None

    with open(output_file, 'a', encoding='utf-8') as f:
        for event in data_list:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')

    print(f"📡 抓取 {total} 筆資料，寫入 {output_file}")
    return data_list
