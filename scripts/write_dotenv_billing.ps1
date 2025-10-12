param(
  [string]$PriceMapPath = "price_map.json",
  [string]$EnvPath = ".env"
)

Write-Host "Updating $EnvPath with billing config…" -ForegroundColor Cyan

if (-not (Test-Path -Path $PriceMapPath)) {
  Write-Error "File not found: $PriceMapPath"
  exit 1
}

try {
  $json = Get-Content -Raw -Path $PriceMapPath | ConvertFrom-Json
} catch {
  Write-Error ("Failed to parse JSON from {0}: {1}" -f ${PriceMapPath}, $_)
  exit 2
}

$pairs = @()
foreach ($prop in $json.PSObject.Properties) {
  $pairs += ('{0}:{1}' -f $prop.Name, $prop.Value)
}
$mapString = ($pairs -join ';')

# Gather values
$stripeKey = $Env:STRIPE_SECRET_KEY
if (-not $stripeKey) { $stripeKey = $Env:STRIPE_API_KEY }

if (-not (Test-Path -Path $EnvPath)) {
  New-Item -ItemType File -Path $EnvPath | Out-Null
}

Add-Content -Path $EnvPath -Value "`n# ----- Oscillink billing config -----"
if ($stripeKey) {
  Add-Content -Path $EnvPath -Value ("STRIPE_SECRET_KEY=" + $stripeKey)
} else {
  Write-Warning "STRIPE_SECRET_KEY/API_KEY not set in session; added other values only."
}
Add-Content -Path $EnvPath -Value ("OSCILLINK_STRIPE_PRICE_MAP=" + $mapString)
Add-Content -Path $EnvPath -Value "OSCILLINK_WEBHOOK_EVENTS_COLLECTION=oscillink_webhooks"
Add-Content -Path $EnvPath -Value "OSCILLINK_CUSTOMERS_COLLECTION=oscillink_customers"
Add-Content -Path $EnvPath -Value "OSCILLINK_MONTHLY_USAGE_COLLECTION=oscillink_monthly_usage"
Add-Content -Path $EnvPath -Value "OSCILLINK_STRIPE_MAX_AGE=300"
Add-Content -Path $EnvPath -Value "OSCILLINK_ALLOW_UNVERIFIED_STRIPE=0"

# Load STRIPE_WEBHOOK_SECRET from .env (last entry wins)
$whLine = Get-Content -Path $EnvPath | Select-String -Pattern '^STRIPE_WEBHOOK_SECRET=' | Select-Object -Last 1
if ($whLine) {
  $secret = ($whLine -replace '^STRIPE_WEBHOOK_SECRET=', '')
  if ($secret) {
    $Env:STRIPE_WEBHOOK_SECRET = $secret
    Write-Host "Set STRIPE_WEBHOOK_SECRET in current session." -ForegroundColor Green
  }
} else {
  Write-Warning "STRIPE_WEBHOOK_SECRET not found in $EnvPath; run scripts/stripe_create_webhook.py to create one."
}

Write-Host "Updated $EnvPath and exported OSCILLINK_STRIPE_PRICE_MAP." -ForegroundColor Green
