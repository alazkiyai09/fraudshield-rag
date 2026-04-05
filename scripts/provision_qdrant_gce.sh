#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-your-project-id}"
REGION="${REGION:-asia-southeast1}"
ZONE="${ZONE:-asia-southeast1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-fraudshield-qdrant-vm}"
ADDRESS_NAME="${ADDRESS_NAME:-fraudshield-qdrant-ip}"
PRIVATE_FIREWALL_RULE="${PRIVATE_FIREWALL_RULE:-allow-fraudshield-qdrant-private}"
SERVERLESS_CONNECTOR_RANGE="${SERVERLESS_CONNECTOR_RANGE:-10.8.0.0/28}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-small}"
DATA_DISK_NAME="${DATA_DISK_NAME:-fraudshield-qdrant-data}"
DATA_DISK_SIZE="${DATA_DISK_SIZE:-20GB}"
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:latest}"
QDRANT_API_KEY="${QDRANT_API_KEY:-}"

if [[ "${PROJECT_ID}" == "your-project-id" ]]; then
  echo "Set PROJECT_ID env var before running this script."
  exit 1
fi

if [[ -z "${QDRANT_API_KEY}" ]]; then
  echo "Set QDRANT_API_KEY env var before running this script."
  exit 1
fi

ADDRESS="$(gcloud compute addresses describe "${ADDRESS_NAME}" --region "${REGION}" --format='value(address)')"

gcloud compute firewall-rules describe "${PRIVATE_FIREWALL_RULE}" >/dev/null 2>&1 || \
  gcloud compute firewall-rules create "${PRIVATE_FIREWALL_RULE}" \
    --network=default \
    --allow=tcp:6333 \
    --source-ranges="${SERVERLESS_CONNECTOR_RANGE}" \
    --target-tags=fraudshield-qdrant

cat >/tmp/fraudshield-qdrant-startup.sh <<EOF
#!/usr/bin/env bash
set -euxo pipefail

apt-get update
apt-get install -y docker.io
systemctl enable --now docker

mkdir -p /mnt/disks/qdrant-data

BOOT_DISK_BASE="\$(findmnt -n -o SOURCE / | sed 's/[0-9]*$//')"
DATA_DISK_DEVICE="\$(lsblk -ndo NAME,TYPE | awk '\$2 == \"disk\" { print \"/dev/\" \$1 }' | grep -vx \"\${BOOT_DISK_BASE}\" | head -n 1)"

if [[ -z "\${DATA_DISK_DEVICE}" ]]; then
  echo "Unable to determine attached data disk."
  exit 1
fi

if ! blkid "\${DATA_DISK_DEVICE}" >/dev/null 2>&1; then
  mkfs.ext4 -F "\${DATA_DISK_DEVICE}"
fi

grep -q "/mnt/disks/qdrant-data" /etc/fstab || echo "\${DATA_DISK_DEVICE} /mnt/disks/qdrant-data ext4 defaults,nofail 0 2" >> /etc/fstab
mount -a
chown -R 1000:1000 /mnt/disks/qdrant-data

docker rm -f qdrant caddy || true
docker network create fraudshield-qdrant >/dev/null 2>&1 || true

docker run -d \
  --name qdrant \
  --restart unless-stopped \
  --network fraudshield-qdrant \
  -p 6333:6333 \
  -v /mnt/disks/qdrant-data:/qdrant/storage \
  -e QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY} \
  ${QDRANT_IMAGE}
EOF

if gcloud compute instances describe "${INSTANCE_NAME}" --zone "${ZONE}" >/dev/null 2>&1; then
  gcloud compute instances delete "${INSTANCE_NAME}" --zone "${ZONE}" --quiet
fi

gcloud compute instances create "${INSTANCE_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --machine-type "${MACHINE_TYPE}" \
  --subnet default \
  --address "${ADDRESS}" \
  --tags fraudshield-qdrant \
  --image-family debian-12 \
  --image-project debian-cloud \
  --boot-disk-size 20GB \
  --create-disk "name=${DATA_DISK_NAME},size=${DATA_DISK_SIZE},type=pd-balanced,auto-delete=no" \
  --metadata-from-file startup-script=/tmp/fraudshield-qdrant-startup.sh

INTERNAL_IP="$(gcloud compute instances describe "${INSTANCE_NAME}" --zone "${ZONE}" --format='value(networkInterfaces[0].networkIP)')"
echo "Qdrant private endpoint: http://${INTERNAL_IP}:6333"
