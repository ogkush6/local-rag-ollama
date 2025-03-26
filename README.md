# Local RAG with Ollama

A Retrieval-Augmented Generation (RAG) application that uses local language models via Ollama. This application allows you to:
- Chat with documents by uploading PDF files
- Get answers derived only from the content of your documents
- Configure retrieval parameters for better results

## Table of Contents
---
1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Docker Setup](#docker-setup)
4. [Launch](#launch)
5. [Common Issues and Troubleshooting](#common-issues-and-troubleshooting)
6. [Examples](#examples)
7. [Multilingual Support](#multilingual-support)
---

## Requirements

- [Ollama](https://ollama.ai/download) (for local LLM usage)
- Python 3.9+ 
- PyMuPDF (for PDF document loading)
- FAISS (for vector storage)
- Docker and Docker Compose (for containerized setup)

## Setup

1. **Install Ollama**
   - Download and install [Ollama](https://ollama.ai/download) for your OS
   - Verify installation with: `ollama help`

2. **Download a model**
   - Run: `ollama pull deepseek-r1:7b` (or another compatible model)
   - Wait for the model to download

3. **Setup Python environment**
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/local-rag-ollama.git
   cd local-rag-ollama

   # Create and activate virtual environment
   python -m venv venv
   
   # For Linux/Mac
   source venv/bin/activate
   
   # For Windows
   .\venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

## Docker Setup

This application can be easily deployed using Docker and Docker Compose:

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/local-rag-ollama.git
   cd local-rag-ollama
   ```

2. **Using helper scripts (recommended)**
   - For Linux/Mac:
     ```bash
     # Make the script executable
     chmod +x docker-start.sh
     
     # Run the helper script to build, start and pull the model
     ./docker-start.sh
     ```
   
   - For Windows (PowerShell):
     ```powershell
     # Run the PowerShell helper script
     .\docker-start.ps1
     ```
     
   These scripts will:
   - Check if Docker is installed and running
   - Build and start the containers
   - Pull the necessary model if it doesn't exist
   - Provide instructions for accessing the application

3. **Manual setup**
   - Build and start containers:
     ```bash
     docker-compose up -d
     ```
     
   - Pull the model:
     ```bash
     docker-compose exec ollama ollama pull deepseek-r1:7b
     ```

4. **Access the application**
   - Open your browser and navigate to http://localhost:8000

### Windows Users

If you're using Windows, here are some specific tips:

- Ensure Docker Desktop for Windows is installed and running
- You may need to enable WSL2 (Windows Subsystem for Linux) during Docker Desktop installation
- If using the default CMD or PowerShell terminal, commands should work the same as shown above
- For file paths in volumes, you may need to use Windows-style paths with Docker Desktop

### Docker Configuration Notes:

- The `docker-compose.yml` includes:
  - An Ollama service that runs the language model
  - The RAG application service connected to Ollama
  - GPU support for Ollama if available
  - Health checks for both services
  - Persistent volume for Ollama models

- For GPU support:
  - Ensure NVIDIA Container Toolkit is installed
  - For Windows, use NVIDIA Container Runtime with Docker Desktop
  - The Docker Compose configuration automatically detects and uses available GPUs

- Environment variables:
  - `OLLAMA_HOST`: Set to "ollama" (the service name) for inter-container communication
  - `PORT`: Application port (default is 8000)

## Launch

1. **Start Ollama service**
   ```bash
   # In a separate terminal
   ollama serve
   ```

2. **Start the Chainlit application**
   ```bash
   # Basic usage
   chainlit run app.py
   
   # With custom port
   chainlit run app.py --port 8080
   ```

3. **Access the application**
   - Open your browser and navigate to: http://localhost:8000 (or your custom port)

## Common Issues and Troubleshooting

### PyMuPDF Installation Issues

If you encounter errors related to PDF loading:

```bash
# Install PyMuPDF separately
pip install pymupdf==1.23.21
```

### Embedding Dimension Mismatch

If you see errors about dimension mismatch:

1. This happens when the FAISS vector store was created with a different embedding model than currently used
2. Select "Empty db" when starting the application to create a fresh database
3. Adjust the "How similar should the pieces be?" slider to a lower value (around 0.1) to handle potential negative scores

### Negative Relevance Scores

Ollama embeddings can sometimes produce negative similarity scores. The application has been updated to handle this by:

1. Setting a much lower score threshold (0.1 instead of 0.5)
2. Using proper error handling to catch and recover from issues

### Model Loading Issues

If Ollama fails to load the model:

1. Ensure Ollama is running (`ollama serve` or Docker container is up)
2. Verify the model is downloaded (`ollama list` or via Docker: `docker-compose exec ollama ollama list`)
3. Try a different model if needed (adjust in app.py)

### Docker Issues

1. **Cannot connect to Ollama from app container**:
   - Check if the Ollama container is healthy: `docker-compose ps`
   - Verify the model is downloaded: `docker-compose exec ollama ollama list`
   - Check logs: `docker-compose logs ollama`

2. **Application container fails to start**:
   - Check logs: `docker-compose logs rag-app`
   - Ensure Ollama container is running first
   - Verify network connectivity between containers

## Examples
---
![ML answer](https://github.com/sidjik/local-rag-ollama/blob/main/imgs/MLAnswer.JPG)
![Example of ML answer clue](https://github.com/sidjik/local-rag-ollama/blob/main/imgs/MLDoc.JPG)
---
![Linear Regression in python answer](https://github.com/sidjik/local-rag-ollama/blob/main/imgs/linearRegression.JPG)
![Linear Regression in python answer clue](https://github.com/sidjik/local-rag-ollama/blob/main/imgs/reggressionLinear.JPG)
---

## Multilingual Support

This application includes a novel approach to handling non-English queries:

1. When a non-English query is received, the LLM generates a potential answer (hallucination)
2. This hallucination is then used to search the vector database for relevant document sections
3. If relevant sections are found, they are used to generate a proper response

This technique allows the application to work with queries in languages other than English, even when the embedding model only supports English.

![Query scheme](https://github.com/sidjik/local-rag-ollama/blob/main/imgs/queryScheme.jpg)

Example with a Slovak question:
![Slovak question example](https://github.com/sidjik/local-rag-ollama/blob/main/imgs/querySchemeExample.jpg)


