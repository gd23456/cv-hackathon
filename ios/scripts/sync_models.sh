#!/bin/sh
# Copies the compiled Core ML models into the app bundle's resources.
# Usage: ios/scripts/sync_models.sh SOURCE_DIR
# SOURCE_DIR is the compiled folder written by optimize/convert_coreml.py: the 8-bit generator, variant A.
set -eu
SRC=${1:?usage: sync_models.sh SOURCE_DIR}
DST="$(cd "$(dirname "$0")/.." && pwd)/App/Resources/Models"
mkdir -p "$DST"
rm -rf "$DST/TextEncoder.mlmodelc" "$DST/UNet.mlmodelc" "$DST/TAESD.mlmodelc"
cp -RL "$SRC/text_pal8.mlmodelc" "$DST/TextEncoder.mlmodelc"
cp -RL "$SRC/unet_fix_pal8.mlmodelc" "$DST/UNet.mlmodelc"
cp -RL "$SRC/taesd_fp16.mlmodelc" "$DST/TAESD.mlmodelc"
du -sh "$DST"/*
