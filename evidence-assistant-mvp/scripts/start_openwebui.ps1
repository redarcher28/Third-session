# 启动证据助手后端 (8000) + Open WebUI 主前端 (8080)
# 用法：在项目根目录 PowerShell 执行  .\scripts\start_openwebui.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$BackendHost = if ($env:EVIDENCE_BACKEND_HOST) { $env:EVIDENCE_BACKEND_HOST } else { "127.0.0.1" }
$BackendPort = if ($env:EVIDENCE_BACKEND_PORT) { $env:EVIDENCE_BACKEND_PORT } else { "8000" }
$OpenWebUIHost = if ($env:OPENWEBUI_HOST) { $env:OPENWEBUI_HOST } else { "127.0.0.1" }
$OpenWebUIPort = if ($env:OPENWEBUI_PORT) { $env:OPENWEBUI_PORT } else { "8080" }
$OpenWebUIName = if ($env:OPENWEBUI_NAME) { $env:OPENWEBUI_NAME } else { "证据台" }
$DataDir = if ($env:OPENWEBUI_DATA_DIR) {
    $env:OPENWEBUI_DATA_DIR
} else {
    Join-Path $env:LOCALAPPDATA "evidence-assistant-mvp\openwebui"
}
$SettingsUrl = "http://${BackendHost}:${BackendPort}/settings"

function Find-Python {
    if ($env:EVIDENCE_PYTHON -and (Test-Path $env:EVIDENCE_PYTHON)) { return $env:EVIDENCE_PYTHON }
    $venvPy = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPy) { return $venvPy }
    return "python"
}

function Find-OpenWebUIPython {
    if ($env:OPENWEBUI_PYTHON -and (Test-Path $env:OPENWEBUI_PYTHON)) { return $env:OPENWEBUI_PYTHON }
    $condaPy = "D:\conda\envs\open_webui\python.exe"
    if (Test-Path $condaPy) { return $condaPy }
    return Find-Python
}

$Python = Find-Python
$OpenWebUIPython = Find-OpenWebUIPython
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

Write-Host "============================================================"
Write-Host "  Evidence Assistant · 后端 + Open WebUI"
Write-Host "  后端 API   http://${BackendHost}:${BackendPort}/"
Write-Host "  Open WebUI http://${OpenWebUIHost}:${OpenWebUIPort}/  ($OpenWebUIName)"
Write-Host "  自定义站点 http://${BackendHost}:${BackendPort}/consult"
Write-Host "  备用页     http://${BackendHost}:${BackendPort}/fallback"
Write-Host "  按 Ctrl+C 停止"
Write-Host "============================================================"

$backend = Start-Process -FilePath $Python -ArgumentList @(
    "-m", "uvicorn", "src.app.api:app",
    "--host", $BackendHost, "--port", $BackendPort
) -PassThru -WorkingDirectory $ProjectRoot -WindowStyle Hidden

Start-Sleep -Seconds 2
for ($i = 0; $i -lt 30; $i++) {
    try {
        Invoke-WebRequest -Uri "http://${BackendHost}:${BackendPort}/health" -UseBasicParsing -TimeoutSec 2 | Out-Null
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}

Write-Host "安装 Open WebUI 证据模型设置桥接..."
& $OpenWebUIPython (Join-Path $ProjectRoot "scripts\install_openwebui_bridge.py") `
    --backend-url "http://${BackendHost}:${BackendPort}" 2>$null

$secretFile = Join-Path $DataDir ".webui_secret_key"
if ((Test-Path $secretFile) -and ((Get-Item $secretFile).Length -gt 0)) {
    $secret = Get-Content $secretFile -Raw
} else {
    $secret = & $OpenWebUIPython -c "import secrets; print(secrets.token_urlsafe(48))"
    Set-Content -Path $secretFile -Value $secret -NoNewline
}

$env:WEBUI_NAME = $OpenWebUIName
$env:DEFAULT_LOCALE = if ($env:DEFAULT_LOCALE) { $env:DEFAULT_LOCALE } else { "zh-CN" }
$env:ENABLE_EVALUATION_ARENA_MODELS = "false"
$env:OPENAI_API_BASE_URL = "http://${BackendHost}:${BackendPort}/v1"
$env:OPENAI_API_BASE_URLS = "http://${BackendHost}:${BackendPort}/v1"
$env:OPENAI_API_KEY = "evidence-local"
$env:OPENAI_API_KEYS = "evidence-local"
$env:DATA_DIR = $DataDir
$env:WEBUI_SECRET_KEY = $secret
$env:RAG_EMBEDDING_ENGINE = "openai"
$env:RAG_EMBEDDING_MODEL = "evidence-embedding"
$env:BYPASS_EMBEDDING_AND_RETRIEVAL = "true"

$openwebui = Start-Process -FilePath (Join-Path (Split-Path $OpenWebUIPython -Parent) "Scripts\open-webui.exe") -ArgumentList @(
    "serve",
    "--host", $OpenWebUIHost, "--port", $OpenWebUIPort
) -PassThru -WorkingDirectory $ProjectRoot -WindowStyle Hidden

Start-Sleep -Seconds 3
for ($i = 0; $i -lt 30; $i++) {
    try {
        Invoke-WebRequest -Uri "http://${OpenWebUIHost}:${OpenWebUIPort}/api/config" -UseBasicParsing -TimeoutSec 2 | Out-Null
        & $OpenWebUIPython (Join-Path $ProjectRoot "scripts\configure_openwebui.py") `
            --data-dir $DataDir --settings-url $SettingsUrl 2>$null
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}

Write-Host "主界面：http://${OpenWebUIHost}:${OpenWebUIPort}/"
Write-Host "自定义咨询页：http://${BackendHost}:${BackendPort}/consult"

try {
    Wait-Process -Id $backend.Id, $openwebui.Id
} finally {
    if (-not $backend.HasExited) { Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue }
    if (-not $openwebui.HasExited) { Stop-Process -Id $openwebui.Id -Force -ErrorAction SilentlyContinue }
}
