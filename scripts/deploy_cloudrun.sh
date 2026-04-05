#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-your-project-id}"
REGION="${REGION:-asia-southeast1}"
SERVICE_NAME="${SERVICE_NAME:-fraudshield-api}"
ALLOW_UNAUTHENTICATED="${ALLOW_UNAUTHENTICATED:-true}"
LLM_PROVIDER="${LLM_PROVIDER:-anthropic}"
LLM_MODEL="${LLM_MODEL:-glm-5.1}"
ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://api.z.ai/api/anthropic}"
ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-http://localhost,http://127.0.0.1}"
QDRANT_MODE="${QDRANT_MODE:-local}"
QDRANT_PATH="${QDRANT_PATH:-/app/data/qdrant}"
QDRANT_URL="${QDRANT_URL:-}"
QDRANT_HOST="${QDRANT_HOST:-localhost}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
QDRANT_HTTPS="${QDRANT_HTTPS:-false}"
QDRANT_API_KEY_SECRET="${QDRANT_API_KEY_SECRET:-}"
API_KEY_SECRET="${API_KEY_SECRET:-service-api-key}"
ANTHROPIC_API_KEY_SECRET="${ANTHROPIC_API_KEY_SECRET:-glm-api-key}"
VPC_CONNECTOR="${VPC_CONNECTOR:-}"
VPC_EGRESS="${VPC_EGRESS:-private-ranges-only}"
ENV_VARS="^@^LLM_PROVIDER=${LLM_PROVIDER}@LLM_MODEL=${LLM_MODEL}@ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL}@ALLOWED_ORIGINS=${ALLOWED_ORIGINS}@QDRANT_MODE=${QDRANT_MODE}@QDRANT_PATH=${QDRANT_PATH}@QDRANT_URL=${QDRANT_URL}@QDRANT_HOST=${QDRANT_HOST}@QDRANT_PORT=${QDRANT_PORT}@QDRANT_HTTPS=${QDRANT_HTTPS}"
UPDATE_SECRETS="API_KEY=${API_KEY_SECRET}:latest,ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY_SECRET}:latest"

if [[ -n "${QDRANT_API_KEY_SECRET}" ]]; then
  UPDATE_SECRETS="${UPDATE_SECRETS},QDRANT_API_KEY=${QDRANT_API_KEY_SECRET}:latest"
fi

if [[ "${PROJECT_ID}" == "your-project-id" ]]; then
  echo "Set PROJECT_ID env var before running this script."
  exit 1
fi

deploy_args=(
  "${SERVICE_NAME}"
  --project "${PROJECT_ID}"
  --source .
  --region "${REGION}"
  --platform managed
  --memory 2Gi
  --cpu 1
  --min-instances 0
  --max-instances 3
  --timeout 300
  --concurrency 80
  --set-env-vars "${ENV_VARS}"
  --update-secrets "${UPDATE_SECRETS}"
)

if [[ "${ALLOW_UNAUTHENTICATED,,}" == "true" ]]; then
  deploy_args+=(--allow-unauthenticated)
fi

if [[ -n "${VPC_CONNECTOR}" ]]; then
  deploy_args+=(--vpc-connector "${VPC_CONNECTOR}" --vpc-egress "${VPC_EGRESS}")
fi

gcloud run deploy "${deploy_args[@]}"

echo "Deployed URL:"
gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format 'value(status.url)'
