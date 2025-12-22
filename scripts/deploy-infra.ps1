# Deploy Azure Infrastructure for Advanced Agentic RAG
# Run this script once to provision all Azure resources

param(
    [string]$ResourceGroup = "rg-agentic-rag",
    [string]$Location = "eastus",
    [string]$OpenAIApiKey = ""
)

$ErrorActionPreference = "Stop"

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "Deploying Azure Infrastructure" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Check Azure CLI login
Write-Host "Checking Azure CLI login..." -ForegroundColor Yellow
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account) {
    Write-Host "Please login to Azure CLI first: az login" -ForegroundColor Red
    exit 1
}
Write-Host "Logged in as: $($account.user.name)" -ForegroundColor Green
Write-Host "Subscription: $($account.name)" -ForegroundColor Green
Write-Host ""

# Check resource group exists
Write-Host "Checking resource group..." -ForegroundColor Yellow
$rg = az group show --name $ResourceGroup 2>$null | ConvertFrom-Json
if (-not $rg) {
    Write-Host "Creating resource group: $ResourceGroup in $Location" -ForegroundColor Yellow
    az group create --name $ResourceGroup --location $Location
} else {
    Write-Host "Resource group exists: $ResourceGroup" -ForegroundColor Green
}
Write-Host ""

# Deploy Bicep template
Write-Host "Deploying infrastructure (this may take 5-10 minutes)..." -ForegroundColor Yellow
$deployParams = @(
    "deployment", "group", "create",
    "--resource-group", $ResourceGroup,
    "--template-file", "infra/main.bicep",
    "--parameters", "infra/parameters.json"
)

if ($OpenAIApiKey) {
    $deployParams += "--parameters"
    $deployParams += "openaiApiKey=$OpenAIApiKey"
}

$result = az @deployParams | ConvertFrom-Json

if ($LASTEXITCODE -ne 0) {
    Write-Host "Deployment failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "====================================" -ForegroundColor Green
Write-Host "Deployment Successful!" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host ""

# Extract outputs
$outputs = $result.properties.outputs
Write-Host "Resources Created:" -ForegroundColor Cyan
Write-Host "  ACR Login Server: $($outputs.acrLoginServer.value)"
Write-Host "  ACR Name: $($outputs.acrName.value)"
Write-Host "  Container App URL: $($outputs.acaAppUrl.value)"
Write-Host "  Key Vault: $($outputs.keyVaultName.value)"
Write-Host "  App Insights: $($outputs.appInsightsName.value)"
Write-Host ""

# Get ACR credentials for GitHub secrets
Write-Host "ACR Credentials (for GitHub Secrets):" -ForegroundColor Yellow
$acrCreds = az acr credential show --name $outputs.acrName.value | ConvertFrom-Json
Write-Host "  ACR_USERNAME: $($acrCreds.username)"
Write-Host "  ACR_PASSWORD: $($acrCreds.passwords[0].value)"
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. Add these GitHub secrets to your repository:"
Write-Host "   - AZURE_CREDENTIALS (service principal JSON)"
Write-Host "   - ACR_USERNAME"
Write-Host "   - ACR_PASSWORD"
Write-Host ""
Write-Host "2. Create Azure service principal for GitHub Actions:"
Write-Host "   az ad sp create-for-rbac --name 'github-agentic-rag' --role contributor --scopes /subscriptions/<sub-id>/resourceGroups/$ResourceGroup --sdk-auth"
Write-Host ""
Write-Host "3. If you didn't provide OpenAI API key, add it to Key Vault:"
Write-Host "   az keyvault secret set --vault-name $($outputs.keyVaultName.value) --name OPENAI-API-KEY --value '<your-key>'"
Write-Host ""
Write-Host "4. Push to main branch to trigger deployment"
