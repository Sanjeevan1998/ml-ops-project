#!/bin/bash

# === Configuration ===
TACC_USER="<your_username>"
TACC_SERVER="<server_ip>"
TACC_MOUNT="/mnt/persistent_storage/raw_pdfs/"
MAC_CASLAW_DIR="~/Desktop/CaseLaw/"
RCLONE_CONTAINER="<your_bucket_name>"

# === Sync Local Files to TACC ===
echo "🚀 Syncing CaseLaw from Mac to TACC..."

rsync -avh --progress $MAC_CASLAW_DIR $TACC_USER@$TACC_SERVER:$TACC_MOUNT

if [ $? -ne 0 ]; then
    echo "❌ Error syncing files to TACC. Exiting."
    exit 1
fi

echo "✅ Sync complete. Files now on TACC node."

# === Run Docker Compose on TACC to Upload to Object Store ===
echo "🚀 Running rclone container on TACC..."

ssh $TACC_USER@$TACC_SERVER << EOF
export RCLONE_CONTAINER=$RCLONE_CONTAINER

docker-compose -f /path/to/docker-compose-upload.yaml up

echo "✅ Data uploaded to Object Store."
EOF

echo "✅ Done! Your CaseLaw files are now in the object store."
