#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Local RAG with Ollama in Docker...${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker and try again.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose and try again.${NC}"
    exit 1
fi

# Build and start containers
echo -e "${YELLOW}Building and starting containers...${NC}"
docker-compose up -d

# Wait for Ollama container to be ready
echo -e "${YELLOW}Waiting for Ollama container to be ready...${NC}"
sleep 10

# Check if model exists, pull if not
echo -e "${YELLOW}Checking for deepseek-r1:7b model...${NC}"
if ! docker-compose exec -T ollama ollama list | grep -q "deepseek-r1:7b"; then
    echo -e "${YELLOW}Model not found. Pulling deepseek-r1:7b model (this may take a while)...${NC}"
    docker-compose exec -T ollama ollama pull deepseek-r1:7b
else
    echo -e "${GREEN}Model deepseek-r1:7b already exists.${NC}"
fi

# Print access instructions
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}Local RAG with Ollama is now running in Docker!${NC}"
echo -e "${GREEN}Access the application at: http://localhost:8000${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""
echo -e "To view logs: ${YELLOW}docker-compose logs -f${NC}"
echo -e "To stop: ${YELLOW}docker-compose down${NC}" 