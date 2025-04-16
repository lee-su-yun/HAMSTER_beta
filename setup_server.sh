#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Store IP address
ifconfig eth0 | grep 'inet ' | awk '{print $2}' > ip_eth0.txt

# Clone the model from Hugging Face
git lfs install
git clone https://huggingface.co/yili18/Hamster_dev

# Run our custom server
python -W ignore server.py \
    --port 8000 \
    --model-path Hamster_dev \
    --conv-mode vicuna_v1
