#!/bin/bash

# Install uv
pip install uv

# Use system
export UV_SYSTEM_PYTHON=1

# Install PyTorch CPU version
uv pip install --index-url https://download.pytorch.org/whl/cpu torch

# Install ML service dependencies
uv pip install -e ./services/meridian-ml-service/.[dev]

# Create models directory
mkdir -p ./services/meridian-ml-service/models

# Download the multilingual-e5-small model
echo "Downloading intfloat/multilingual-e5-small model..."
python -c "
import os
from transformers import AutoModel, AutoTokenizer

model_name = 'intfloat/multilingual-e5-small'
models_dir = './services/meridian-ml-service/models'

print(f'Downloading {model_name} to {models_dir}...')

# Download tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Save to local models directory
tokenizer.save_pretrained(models_dir)
model.save_pretrained(models_dir)

print(f'Model successfully downloaded to {models_dir}')
"

# Update .env file with local model path
MODEL_FULL_PATH="$(pwd)/services/meridian-ml-service/models"

cd ./services/meridian-ml-service

if [ -f .env ]; then
    # Update existing EMBEDDING_MODEL_NAME or add it
    if grep -q "EMBEDDING_MODEL_NAME=" .env; then
        sed -i "s|EMBEDDING_MODEL_NAME=.*|EMBEDDING_MODEL_NAME=$MODEL_FULL_PATH|" .env
    else
        echo "EMBEDDING_MODEL_NAME=$MODEL_FULL_PATH" >> .env
    fi
else
    # Create .env from example and update it
    cp .env.example .env
    sed -i "s|EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-small|EMBEDDING_MODEL_NAME=$MODEL_FULL_PATH|" .env
fi

echo "Setup complete! Model downloaded and .env configured."

cd ../..
