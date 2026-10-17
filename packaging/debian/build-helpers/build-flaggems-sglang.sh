#!/usr/bin/env bash
# Build python3-flaggems-sglang_*.deb locally.
#   usage: build-flaggems-sglang.sh [ubuntu-version]   (default 24.04)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

BASE_IMAGE_VERSION="${1:-24.04}"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile.deb"
IMAGE_TAG="flaggems-sglang-deb:${BASE_IMAGE_VERSION}"
OUTPUT_DIR="${PROJECT_DIR}/debian-packages"

docker build --network=host \
    -f "$DOCKERFILE" \
    --build-arg BASE_IMAGE_VERSION="$BASE_IMAGE_VERSION" \
    -t "$IMAGE_TAG" \
    "$PROJECT_DIR"

mkdir -p "$OUTPUT_DIR"
CONTAINER_ID=$(docker create "$IMAGE_TAG")
docker cp "$CONTAINER_ID:/output/." "$OUTPUT_DIR/"
docker rm "$CONTAINER_ID" > /dev/null

echo ""
echo ">>> Output:"
ls -lh "$OUTPUT_DIR"
