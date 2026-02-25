$ErrorActionPreference = "Stop"

Write-Host "Stopping Ollama service..."
docker compose stop ollama | Out-Host
