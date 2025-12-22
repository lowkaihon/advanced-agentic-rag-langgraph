#!/bin/bash
# Deploy Azure Infrastructure for Advanced Agentic RAG
# Run this script once to provision all Azure resources

set -e

RESOURCE_GROUP="${1:-rg-agentic-rag}"
LOCATION="${2:-eastus}"
OPENAI_API_KEY="${3:-}"

echo "===================================="
echo "Deploying Azure Infrastructure"
echo "===================================="
echo ""

# Check Azure CLI login
echo "Checking Azure CLI login..."
if ! az account show > /dev/null 2>&1; then
    echo "Please login to Azure CLI first: az login"
    exit 1
fi
ACCOUNT_NAME=$(az account show --query "user.name" -o tsv)
SUB_NAME=$(az account show --query "name" -o tsv)
echo "Logged in as: $ACCOUNT_NAME"
echo "Subscription: $SUB_NAME"
echo ""

# Check resource group exists
echo "Checking resource group..."
if ! az group show --name "$RESOURCE_GROUP" > /dev/null 2>&1; then
    echo "Creating resource group: $RESOURCE_GROUP in $LOCATION"
    az group create --name "$RESOURCE_GROUP" --location "$LOCATION"
else
    echo "Resource group exists: $RESOURCE_GROUP"
fi
echo ""

# Deploy Bicep template
echo "Deploying infrastructure (this may take 5-10 minutes)..."

DEPLOY_PARAMS="--resource-group $RESOURCE_GROUP --template-file infra/main.bicep --parameters infra/parameters.json"

if [ -n "$OPENAI_API_KEY" ]; then
    DEPLOY_PARAMS="$DEPLOY_PARAMS --parameters openaiApiKey=$OPENAI_API_KEY"
fi

RESULT=$(az deployment group create $DEPLOY_PARAMS -o json)

echo ""
echo "===================================="
echo "Deployment Successful!"
echo "===================================="
echo ""

# Extract outputs
ACR_LOGIN_SERVER=$(echo "$RESULT" | jq -r '.properties.outputs.acrLoginServer.value')
ACR_NAME=$(echo "$RESULT" | jq -r '.properties.outputs.acrName.value')
ACA_URL=$(echo "$RESULT" | jq -r '.properties.outputs.acaAppUrl.value')
KV_NAME=$(echo "$RESULT" | jq -r '.properties.outputs.keyVaultName.value')
APPI_NAME=$(echo "$RESULT" | jq -r '.properties.outputs.appInsightsName.value')

echo "Resources Created:"
echo "  ACR Login Server: $ACR_LOGIN_SERVER"
echo "  ACR Name: $ACR_NAME"
echo "  Container App URL: $ACA_URL"
echo "  Key Vault: $KV_NAME"
echo "  App Insights: $APPI_NAME"
echo ""

# Get ACR credentials for GitHub secrets
echo "ACR Credentials (for GitHub Secrets):"
ACR_CREDS=$(az acr credential show --name "$ACR_NAME" -o json)
ACR_USERNAME=$(echo "$ACR_CREDS" | jq -r '.username')
ACR_PASSWORD=$(echo "$ACR_CREDS" | jq -r '.passwords[0].value')
echo "  ACR_USERNAME: $ACR_USERNAME"
echo "  ACR_PASSWORD: $ACR_PASSWORD"
echo ""

echo "Next Steps:"
echo "1. Add these GitHub secrets to your repository:"
echo "   - AZURE_CREDENTIALS (service principal JSON)"
echo "   - ACR_USERNAME"
echo "   - ACR_PASSWORD"
echo ""
echo "2. Create Azure service principal for GitHub Actions:"
echo "   az ad sp create-for-rbac --name 'github-agentic-rag' --role contributor --scopes /subscriptions/<sub-id>/resourceGroups/$RESOURCE_GROUP --sdk-auth"
echo ""
echo "3. If you didn't provide OpenAI API key, add it to Key Vault:"
echo "   az keyvault secret set --vault-name $KV_NAME --name OPENAI-API-KEY --value '<your-key>'"
echo ""
echo "4. Push to main branch to trigger deployment"
