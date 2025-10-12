# Load price_map.json and set OSCILLINK_STRIPE_PRICE_MAP for current PowerShell session
param(
  [string]$Path = "price_map.json"
)

if (-not (Test-Path -Path $Path)) {
  Write-Error "File not found: $Path"
  exit 1
}

try {
  $json = Get-Content -Raw -Path $Path | ConvertFrom-Json
} catch {
  # Use ${} to avoid ':' parsing issues in interpolated strings
  Write-Error ("Failed to parse JSON from {0}: {1}" -f ${Path}, $_)
  exit 2
}

# Convert mapping object (price_id -> tier) into semicolon-delimited string: price: tier
$pairs = @()
foreach ($prop in $json.PSObject.Properties) {
  $pairs += ("{0}:{1}" -f $prop.Name, $prop.Value)
}
$mapString = ($pairs -join ';')

$Env:OSCILLINK_STRIPE_PRICE_MAP = $mapString
Write-Host "Set OSCILLINK_STRIPE_PRICE_MAP for this session:" -ForegroundColor Green
Write-Host ("  {0}" -f $Env:OSCILLINK_STRIPE_PRICE_MAP)
