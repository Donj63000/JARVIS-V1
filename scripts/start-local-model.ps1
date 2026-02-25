param(
    [string]$Model = "qwen2.5-coder:14b"
)

$ErrorActionPreference = "Stop"

Write-Host "Starting Ollama service (Docker Compose)..."
docker compose up -d ollama | Out-Host

$ready = $false
for ($attempt = 1; $attempt -le 60; $attempt++) {
    try {
        Invoke-RestMethod -Method Get -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 | Out-Null
        $ready = $true
        break
    }
    catch {
        Start-Sleep -Seconds 2
    }
}

if (-not $ready) {
    throw "Ollama did not respond on http://localhost:11434 after 120 seconds."
}

Write-Host "Pulling model $Model in project volume..."
$payload = @{
    name = $Model
    stream = $false
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://localhost:11434/api/pull" -ContentType "application/json" -Body $payload | Out-Null

Write-Host "Model ready. Launch app with: cargo run --release"
