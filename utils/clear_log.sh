#!/bin/bash
# Clear the contents of the specified JSONL file if it exists
FILE="data/new_attack_data.jsonl"
if [ -f "$FILE" ]; then
    > "$FILE"
else
    echo "File $FILE does not exist."
fi
