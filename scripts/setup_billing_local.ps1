<#
Oscillink local billing setup (Windows PowerShell)
- Prompts for your Stripe keys and Beta price id
- Sets env vars for current PowerShell session
- Optionally runs the dev server on port 8000

Usage:
  scripts\setup_billing_local.ps1

You can re-run anytime; it will overwrite the current session env vars only.
#>
param(
  [switch]$RunServer
)

Write-Host "Oscillink — local billing setup" -ForegroundColor Cyan

# Prompt helpers
function Prompt-Secret($name) {
  $val = Read-Host -AsSecureString $name
  if (-not $val) { return $null }
  $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($val)
  $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
  [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
  return $plain
}

# Collect inputs
$existingStripe = $Env:STRIPE_SECRET_KEY
if (-not $existingStripe) {
  $stripe = Prompt-Secret "Enter STRIPE_SECRET_KEY (sk_live_...)"
} else {
  Write-Host "Found STRIPE_SECRET_KEY in session; press Enter to keep or paste new." -ForegroundColor Yellow
  $stripe = Prompt-Secret "Enter STRIPE_SECRET_KEY (sk_live_...)"
  if (-not $stripe) { $stripe = $existingStripe }
}

$existingWebhook = $Env:STRIPE_WEBHOOK_SECRET
if (-not $existingWebhook) {
  $wh = Read-Host "Enter STRIPE_WEBHOOK_SECRET (whsec_..., optional; Enter to skip)"
} else {
  Write-Host "Found STRIPE_WEBHOOK_SECRET in session; press Enter to keep or paste new." -ForegroundColor Yellow
  $tmp = Read-Host "Enter STRIPE_WEBHOOK_SECRET (whsec_..., optional; Enter to keep)"
  if ($tmp) { $wh = $tmp } else { $wh = $existingWebhook }
}

$existingMap = $Env:OSCILLINK_STRIPE_PRICE_MAP
Write-Host "Set OSCILLINK_STRIPE_PRICE_MAP so the server can map your price id to the 'beta' tier." -ForegroundColor Yellow
Write-Host "Format: price_xxx:beta or JSON like {\"price_xxx\":\"beta\"}" -ForegroundColor DarkGray
if (-not $existingMap) {
  $map = Read-Host "Enter price map (e.g., price_123:beta)"
} else {
  Write-Host "Found OSCILLINK_STRIPE_PRICE_MAP in session; press Enter to keep or paste new." -ForegroundColor Yellow
  $tmp = Read-Host "Enter price map (e.g., price_123:beta)"
  if ($tmp) { $map = $tmp } else { $map = $existingMap }
}

# Apply to current session
$Env:STRIPE_SECRET_KEY = $stripe
if ($wh) { $Env:STRIPE_WEBHOOK_SECRET = $wh }
$Env:OSCILLINK_STRIPE_PRICE_MAP = $map

# Recommended developer flags
$Env:OSCILLINK_FORCE_HTTPS = '0'
$Env:OSCILLINK_ALLOW_UNVERIFIED_STRIPE = '1'  # local/dev convenience; do NOT set in prod

Write-Host "\nConfigured environment for this session:" -ForegroundColor Green
Write-Host "  STRIPE_SECRET_KEY: " ($Env:STRIPE_SECRET_KEY.Substring(0,6) + "…")
if ($Env:STRIPE_WEBHOOK_SECRET) { Write-Host "  STRIPE_WEBHOOK_SECRET: " ($Env:STRIPE_WEBHOOK_SECRET.Substring(0,5) + "…") }
Write-Host "  OSCILLINK_STRIPE_PRICE_MAP: $Env:OSCILLINK_STRIPE_PRICE_MAP"
Write-Host "  OSCILLINK_FORCE_HTTPS: $Env:OSCILLINK_FORCE_HTTPS"
Write-Host "  OSCILLINK_ALLOW_UNVERIFIED_STRIPE: $Env:OSCILLINK_ALLOW_UNVERIFIED_STRIPE"

if ($RunServer) {
  Write-Host "\nStarting dev server on http://localhost:8000 ..." -ForegroundColor Cyan
  uvicorn cloud.app.main:app --port 8000
} else {
  Write-Host "\nTip: start the dev server with:" -ForegroundColor Cyan
  Write-Host "  uvicorn cloud.app.main:app --port 8000"
}
