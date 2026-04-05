#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-asia-southeast1}"
SERVICE_NAME="${SERVICE_NAME:-fraudshield-demo}"
API_URL="${API_URL:-https://fraudshield-api-5tphgb6fsa-as.a.run.app}"
API_KEY_SECRET="${API_KEY_SECRET:-service-api-key}"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../demo" && pwd)"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is required."
  exit 1
fi

gcloud run deploy "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --source "${SOURCE_DIR}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 2 \
  --timeout 300 \
  --concurrency 50 \
  --set-env-vars "API_URL=${API_URL}" \
  --update-secrets "API_KEY=${API_KEY_SECRET}:latest"

gcloud run services describe "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --format='value(status.url)'
