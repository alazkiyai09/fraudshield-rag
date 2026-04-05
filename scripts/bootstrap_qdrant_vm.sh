#!/usr/bin/env bash
set -euo pipefail

QDRANT_API_KEY="${QDRANT_API_KEY:?Set QDRANT_API_KEY}"
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:latest}"

mkdir -p /mnt/disks/qdrant-data

BOOT_DISK_BASE="$(findmnt -n -o SOURCE / | sed 's/[0-9]*$//')"
DATA_DISK_DEVICE="$(lsblk -ndo NAME,TYPE | awk '$2 == "disk" { print "/dev/" $1 }' | grep -vx "${BOOT_DISK_BASE}" | head -n 1)"

if [[ -z "${DATA_DISK_DEVICE}" ]]; then
  echo "Unable to determine attached data disk."
  exit 1
fi

if ! blkid "${DATA_DISK_DEVICE}" >/dev/null 2>&1; then
  mkfs.ext4 -F "${DATA_DISK_DEVICE}"
fi

grep -q "/mnt/disks/qdrant-data" /etc/fstab || \
  echo "${DATA_DISK_DEVICE} /mnt/disks/qdrant-data ext4 defaults,nofail 0 2" >> /etc/fstab

mount -a
systemctl enable --now docker

docker rm -f qdrant caddy || true
docker network create fraudshield-qdrant >/dev/null 2>&1 || true

docker run -d \
  --name qdrant \
  --restart unless-stopped \
  --network fraudshield-qdrant \
  -p 6333:6333 \
  -v /mnt/disks/qdrant-data:/qdrant/storage \
  -e QDRANT__SERVICE__API_KEY="${QDRANT_API_KEY}" \
  "${QDRANT_IMAGE}"

docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
