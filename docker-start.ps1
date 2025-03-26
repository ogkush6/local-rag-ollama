# PowerShell script for starting the Local RAG with Ollama in Docker on Windows

# Output colors
$colors = @{
    Green = [System.Console]::ForegroundColor = 'Green'
    Yellow = [System.Console]::ForegroundColor = 'Yellow'
    Red = [System.Console]::ForegroundColor = 'Red'
    Default = [System.Console]::ResetColor()
}

function Write-ColorOutput($color, $message) {
    [System.Console]::ForegroundColor = $color
    Write-Output $message
    [System.Console]::ResetColor()
}

Write-ColorOutput 'Green' "Starting Local RAG with Ollama in Docker..."

# Check if Docker is installed
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-ColorOutput 'Red' "Docker is not installed. Please install Docker Desktop for Windows and try again."
    exit 1
}

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-ColorOutput 'Red' "Docker is not running. Please start Docker Desktop and try again."
    exit 1
}

# Build and start containers
Write-ColorOutput 'Yellow' "Building and starting containers..."
docker-compose up -d

# Wait for Ollama container to be ready
Write-ColorOutput 'Yellow' "Waiting for Ollama container to be ready..."
Start-Sleep -Seconds 10

# Check if model exists, pull if not
Write-ColorOutput 'Yellow' "Checking for deepseek-r1:7b model..."
$modelExists = docker-compose exec -T ollama ollama list | Select-String -Pattern "deepseek-r1:7b" -Quiet

if (-not $modelExists) {
    Write-ColorOutput 'Yellow' "Model not found. Pulling deepseek-r1:7b model (this may take a while)..."
    docker-compose exec -T ollama ollama pull deepseek-r1:7b
} else {
    Write-ColorOutput 'Green' "Model deepseek-r1:7b already exists."
}

# Print access instructions
Write-ColorOutput 'Green' "=================================================="
Write-ColorOutput 'Green' "Local RAG with Ollama is now running in Docker!"
Write-ColorOutput 'Green' "Access the application at: http://localhost:8000"
Write-ColorOutput 'Green' "=================================================="
Write-Output ""
Write-Output "To view logs: docker-compose logs -f"
Write-Output "To stop: docker-compose down" 