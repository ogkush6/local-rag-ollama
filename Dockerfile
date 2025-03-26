# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for PyMuPDF and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Environment variables
ENV OLLAMA_HOST=ollama
ENV PORT=8000

# Create data directory
RUN mkdir -p /app/data

# Expose the application port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"] 