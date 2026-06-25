#!/usr/bin/env bash
set -euo pipefail

# Build the mjlab training image and push it to Baidu CCR.
#
# Usage:
#   CCR_NAMESPACE=mjlabssjj CCR_REPOSITORY=mjlabssjj \
#     scripts/cloud/baidu/build_and_push.sh
#
# Optional:
#   CCR_REGISTRY=registry.baidubce.com
#   IMAGE_TAG=v12-v13
#   PLATFORM=linux/amd64

CCR_REGISTRY="${CCR_REGISTRY:-registry.baidubce.com}"
CCR_NAMESPACE="${CCR_NAMESPACE:?Set CCR_NAMESPACE, e.g. mjlabssjj}"
CCR_REPOSITORY="${CCR_REPOSITORY:?Set CCR_REPOSITORY, e.g. mjlabssjj}"
IMAGE_TAG="${IMAGE_TAG:-v12-v13}"
PLATFORM="${PLATFORM:-linux/amd64}"

IMAGE="${CCR_REGISTRY}/${CCR_NAMESPACE}/${CCR_REPOSITORY}:${IMAGE_TAG}"

echo "Building ${IMAGE}"
docker buildx build \
  --platform "${PLATFORM}" \
  --tag "${IMAGE}" \
  --load \
  .

echo "Pushing ${IMAGE}"
docker push "${IMAGE}"

echo "Image pushed: ${IMAGE}"
