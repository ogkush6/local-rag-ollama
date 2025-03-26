#!/bin/bash
set -e

# Create data directory if it doesn't exist
mkdir -p /app/data

# Wait for Ollama to be ready
echo "Waiting for Ollama service to be ready..."
max_retries=30
counter=0
while ! curl -s http://${OLLAMA_HOST:-ollama}:11434/api/version > /dev/null; do
    sleep 2
    counter=$((counter+1))
    if [ $counter -eq $max_retries ]; then
        echo "Ollama service not available after $max_retries attempts. Proceeding anyway..."
        break
    fi
    echo "Waiting for Ollama service... ($counter/$max_retries)"
done

echo "Starting Chainlit application..."
exec chainlit run app.py --host "0.0.0.0" --port "${PORT:-8000}" 