#!/bin/sh
# Copies the compiled Core ML models into the app bundle's resources.
# Usage: ios/scripts/sync_models.sh SOURCE_DIR
# SOURCE_DIR holds text.mlmodelc, unet.mlmodelc and taesd.mlmodelc (see optimize/).
set -eu
SRC=${1:?usage: sync_models.sh SOURCE_DIR}
DST="$(cd "$(dirname "$0")/.." && pwd)/App/Resources/Models"
mkdir -p "$DST"
rm -rf "$DST/TextEncoder.mlmodelc" "$DST/UNet.mlmodelc" "$DST/TAESD.mlmodelc"
cp -RL "$SRC/text.mlmodelc" "$DST/TextEncoder.mlmodelc"
cp -RL "$SRC/unet.mlmodelc" "$DST/UNet.mlmodelc"
cp -RL "$SRC/taesd.mlmodelc" "$DST/TAESD.mlmodelc"
du -sh "$DST"/*
