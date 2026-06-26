param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$VideoPath = ""
)

$ErrorActionPreference = "Stop"

Write-Host "Health check"
Invoke-RestMethod -Method Get -Uri "$BaseUrl/health" | ConvertTo-Json -Depth 4

Write-Host "`nEye tracking prediction with a generated 80-feature payload"
$featureOrderPath = Join-Path $PSScriptRoot "..\Models\EyeTrackingAssets\feature_order.json"
$featureOrder = Get-Content $featureOrderPath -Raw | ConvertFrom-Json
$features = @{}
foreach ($name in $featureOrder) {
    $features[$name] = 0.0
}

$eyeBody = @{ features = $features } | ConvertTo-Json -Depth 5
Invoke-RestMethod `
    -Method Post `
    -Uri "$BaseUrl/predict/eye-tracking" `
    -ContentType "application/json" `
    -Body $eyeBody | ConvertTo-Json -Depth 4

if ($VideoPath -and (Test-Path -LiteralPath $VideoPath)) {
    Write-Host "`nFacial prediction for $VideoPath"
    curl.exe -s -X POST -F "file=@$VideoPath" "$BaseUrl/predict/facial"
    Write-Host ""
}
else {
    Write-Host "`nSkipping facial upload. Pass -VideoPath path\to\video.mp4 to test /predict/facial."
}
