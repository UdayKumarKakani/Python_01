# AKS Deploy

Required repository secrets:

- `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` — service principal for `az login`.
- `AKS_RESOURCE_GROUP` — resource group containing the AKS cluster.
- `AKS_CLUSTER_NAME` — AKS cluster name.
- `GHCR_USERNAME`, `GHCR_TOKEN` — credentials if pulling from GitHub Container Registry.

Quick usage (from repository root):

```bash
# build & push an image (example)
docker buildx build --platform linux/amd64 --build-arg BASE_IMAGE=python:3.11-slim -t ghcr.io/<OWNER>/<REPO>:latest --push Section_09_Deployment_MLOps/Day_1_Docker_For_AI

# deploy to AKS using the script (make sure secrets are available as env vars locally)
Section_09_Deployment_MLOps/Day_2_CI_CD_GitHub_Actions/scripts/deploy_to_aks.sh ghcr.io/<OWNER>/<REPO>:latest
```

The script logs into Azure using the service principal, fetches AKS credentials and updates or creates the Deployment and Service resources.
