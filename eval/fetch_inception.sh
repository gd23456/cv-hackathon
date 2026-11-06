#!/bin/sh
# clean-fid's Inception model (TorchScript, not safetensors). Run only after the exception is approved.
set -eu
DIR="$HOME/.cache/kshana-eval"
mkdir -p "$DIR"
curl -fsSL -o "$DIR/inception-2015-12-05.pt" https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt
shasum -a 256 "$DIR/inception-2015-12-05.pt"
