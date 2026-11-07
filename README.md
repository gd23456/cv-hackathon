# Kshana

Resource-efficient text-to-image for mobile devices. Computer Vision Hackathon 2026, Problem P3, Team T78.

Results: [`eval_P3_T78.md`](eval_P3_T78.md). Architecture: [`docs/architecture.png`](docs/architecture.png). AI use: [`docs/AI_USAGE.md`](docs/AI_USAGE.md).

## Quickstart

```sh
pip install -r requirements.txt
PYTHONPATH=. python run_inference.py --prompts samples/prompts.txt --output_dir results/samples
```

## Run the website

```sh
conda create -n kshana python=3.11 && conda activate kshana
pip install -r requirements-server.txt torch==2.7.0 diffusers transformers safetensors timm
PYTHONPATH=. python optimize/export_openvino.py --out server/models --backends onnxruntime
PYTHONPATH=. python optimize/export_nsfw.py --onnx server/models/nsfw.onnx
server/run_local.sh
```

Open http://127.0.0.1:8000. To share it publicly, run this in a second terminal:

```sh
cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000
```

Or with Docker: `docker build -t kshana . && docker run -p 8000:8000 kshana`

## Run the iPhone app

Needs a Mac with Xcode, an iPhone in Developer Mode, and an Apple ID added in Xcode.

```sh
brew install xcodegen
PYTHONPATH=. python optimize/convert_coreml.py
ios/scripts/sync_models.sh SOURCE_DIR
ios/scripts/device_run.sh
```

## Reproduce the results

```sh
PYTHONPATH=. python eval/run.py --config S1-A --split coco_5k --inception
sh scripts/finish_eval.sh
PYTHONPATH=. python eval/redteam.py --out results/safety_holdout.csv --benign-skip 300 --benign-n 400 --seeds 55 66 77 88 99
```

## Tests

```sh
PYTHONPATH=. pytest tests/server
cd ios/KshanaCore && swift test
```

## Licences

| Component | Licence |
|---|---|
| SDXS-512-DreamShaper | OpenRAIL++ |
| TAESD | MIT |
| Marqo NSFW classifier | Apache-2.0 |
| YOLOX (server) | Apache-2.0 |
| YOLOv3-Tiny (iPhone) | See pjreddie/darknet |
| MobileCLIP (iPhone) | Apple ASCL |
| Apple BPE tokenizer | MIT |
| COCO, Flickr30K | Used for evaluation only; see their terms |
