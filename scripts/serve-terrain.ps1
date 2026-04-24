param(
    [string]$TilesDir = "tiles",
    [int]$Port = 8080,
    [string]$BindHost = "0.0.0.0"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$CtsDir = Join-Path $RootDir "cts"
$TilesAbs = if ([System.IO.Path]::IsPathRooted($TilesDir)) { $TilesDir } `
    else { Join-Path $RootDir $TilesDir }

if (-not (Test-Path (Join-Path $TilesAbs "layer.json"))) {
    Write-Error "No layer.json found in '$TilesAbs'. Run banin first:`n  cargo run -p banin -- --input <dem.tif> --output $TilesDir"
}

Push-Location $CtsDir
try {
    Write-Host "Building terrain server..." -ForegroundColor Cyan
    go build -o server.exe .

    Write-Host ""
    Write-Host "Serving tiles from : $TilesAbs" -ForegroundColor Green
    Write-Host "Viewer             : http://localhost:$Port/viewer" -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop.`n"

    # Open the viewer in the default browser after a short delay
    Start-Job { Start-Sleep 1; Start-Process "http://localhost:$using:Port/viewer" } | Out-Null

    & "$CtsDir\server.exe" --dir $TilesAbs --port $Port --host $BindHost
}
finally {
    Pop-Location
}
