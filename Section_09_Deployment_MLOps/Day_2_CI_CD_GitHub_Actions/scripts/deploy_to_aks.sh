#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <image>"
  exit 2
fi

IMAGE="$1"

: "${AZURE_CLIENT_ID:?AZURE_CLIENT_ID is required}"
: "${AZURE_CLIENT_SECRET:?AZURE_CLIENT_SECRET is required}"
: "${AZURE_TENANT_ID:?AZURE_TENANT_ID is required}"
: "${AZURE_SUBSCRIPTION_ID:?AZURE_SUBSCRIPTION_ID is required}"
: "${AKS_RESOURCE_GROUP:?AKS_RESOURCE_GROUP is required}"
: "${AKS_CLUSTER_NAME:?AKS_CLUSTER_NAME is required}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
K8S_DIR="$ROOT_DIR/../Day_1_Docker_For_AI/k8s"

echo "Logging into Azure..."
az login --service-principal -u "$AZURE_CLIENT_ID" -p "$AZURE_CLIENT_SECRET" --tenant "$AZURE_TENANT_ID" >/dev/null
az account set --subscription "$AZURE_SUBSCRIPTION_ID"

echo "Fetching AKS credentials for $AKS_CLUSTER_NAME in $AKS_RESOURCE_GROUP..."
az aks get-credentials --resource-group "$AKS_RESOURCE_GROUP" --name "$AKS_CLUSTER_NAME" --overwrite-existing

echo "Deploying image: $IMAGE"
if kubectl get deployment my-app >/dev/null 2>&1; then
  kubectl set image deployment/my-app my-app-container="$IMAGE" --record
else
  sed "s|REPLACE_IMAGE|$IMAGE|g" "$K8S_DIR/deployment.yaml" | kubectl apply -f -
  kubectl apply -f "$K8S_DIR/service.yaml"
fi

kubectl rollout status deployment/my-app

echo "Service endpoint (may take a minute to provision):"
kubectl get svc my-app-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' || kubectl get svc my-app-service -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' || true

echo "Done."
