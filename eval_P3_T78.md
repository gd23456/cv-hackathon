1RUA24CSE0146,1RUA24CSE0201,1RUA24CSE0139,1RUA24CSE0196

## 1. CLIP similarity

Headline ViT-L/14 via torchmetrics, max(100 x cos, 0), against the original prompt, with bootstrap 95% CI. ViT-B/32 is raw cosine.

| Config | Prompts | n | CLIP ViT-L/14 | CLIP ViT-B/32 cosine | Hardware | Engine |
|---|---|---|---|---|---|---|
| S1-A | coco_5k | 5000 | 26.721 [26.632, 26.811] | 0.3122 | Apple M4 | Core ML (coremltools 9.0), CPU_AND_NE |
| S1-A | coco_1k | 1000 | 26.637 [26.438, 26.841] | 0.312 | Apple M4 | Core ML (coremltools 9.0), CPU_AND_NE |
| S1-FP16 | coco_1k | 1000 | 26.643 [26.44, 26.85] | 0.3119 | Apple M4 | Core ML (coremltools 9.0), CPU_AND_NE |
| S0.9-FP16 | coco_1k | 1000 | 26.194 [25.999, 26.388] | 0.3128 | Apple M4 | Core ML (coremltools 9.0), CPU_AND_NE |
| B1 | coco_1k | 1000 | 26.176 [25.983, 26.369] | 0.3147 | Apple M4 | PyTorch 2.7 MPS, fp32 compute |

Flickr30K, separate protocol: official 1k test captions, own real-image reference. Not comparable with the COCO rows.

| Config | Prompts | n | CLIP ViT-L/14 | CLIP ViT-B/32 cosine | Hardware | Engine |
|---|---|---|---|---|---|---|
| S1-A | flickr_1k | 1000 | 27.83 [27.607, 28.06] | 0.3247 | Apple M4 | Core ML (coremltools 9.0), CPU_AND_NE |

iPhone 15: quality is measured on the M4 with the same Core ML models, and wasn't measured separately on the phone.

## 2. FID

clean-fid clean mode, PNG, 5,000 generated vs the same 5,000 COCO val2017 references (center crop, 512 Lanczos).

| Config | n | FID | Hardware | Engine |
|---|---|---|---|---|
| S1-A | 5000 | 32.356 | Apple M4 | Core ML (coremltools 9.0), CPU_AND_NE |

KID and CMMD are for ablations; FID is not reported below 5k. CMMD uses ViT-L/14 at 224, the paper uses 336.

| Config | Split | n | KID x 1000 | CMMD x 1000 |
|---|---|---|---|---|
| S1-A | coco_5k | 5000 | 6.788 | 0.783 |
| S1-A | coco_1k | 1000 | 6.93 | 0.769 |
| S1-FP16 | coco_1k | 1000 | 6.972 | 0.765 |
| S0.9-FP16 | coco_1k | 1000 | 8.808 | 0.643 |
| B1 | coco_1k | 1000 | 6.963 | 0.459 |

Flickr30K, separate protocol: KID against 1,000 Flickr30K test images, preprocessed the same way. No FID below 5k. Not comparable with the COCO rows.

| Config | Split | n | KID x 1000 |
|---|---|---|---|
| S1-A | flickr_1k | 1000 | 12.54 |

## 3. Image generation latency

| Config | Hardware | Engine | Warm n | Mean ms | p50 ms | p95 ms | Text ms | UNet ms | Decoder ms | Cold to first image s |
|---|---|---|---|---|---|---|---|---|---|---|
| S1-A | Apple M4 | Core ML (coremltools 9.0), CPU_AND_NE | 200 | 57.5 | 57.5 | 59.0 | 2.8 | 45.6 | 9.1 | 4.15 |
| S1-FP16 | Apple M4 | Core ML (coremltools 9.0), CPU_AND_NE | 200 | 63.8 | 63.7 | 65.0 | 3.3 | 51.4 | 9.1 | 2.79 |
| S0.9-FP16 | Apple M4 | Core ML (coremltools 9.0), CPU_AND_NE | 200 | 84.4 | 84.2 | 87.5 | 10.5 | 64.7 | 9.2 | 5.3 |
| B1 | Apple M4 | PyTorch 2.7 MPS, fp32 compute | 30 | 1364.8 | 1364.4 | 1391.5 | – | – | – | 8.01 |

| iPhone measure | Value | Device | OS | Config | Conditions | Source |
|---|---|---|---|---|---|---|
| Text ms | about 3 | iPhone 15 | iOS 26.5 | S1-A variant A | warm, 12 runs | app; team lead report 7 Nov 00:12 |
| UNet ms | about 65 | iPhone 15 | iOS 26.5 | S1-A variant A | warm, 12 runs | app; team lead report 7 Nov 00:12 |
| Decoder ms | about 13.5 | iPhone 15 | iOS 26.5 | S1-A variant A | warm, 12 runs | app; team lead report 7 Nov 00:12 |
| Total ms | about 82 | iPhone 15 | iOS 26.5 | S1-A variant A | warm, 12 runs | app; team lead report 7 Nov 00:12 |
| First load s | 66 | iPhone 15 | iOS 26.5 | S1-A variant A | 12 runs batch | app; team lead report 7 Nov 00:12 |
| Cached load s | 20.5 | iPhone 15 | iOS 26.5 | S1-A variant A | 12 runs batch | app; team lead report 7 Nov 00:12 |
| Image model ms | about 67 | iPhone 15 | iOS 26.5 | S1-A variant A | healthy runs, phone hot, thermal state serious | app; team lead update 7 Nov |
| End to end ms | 86-98 | iPhone 15 | iOS 26.5 | S1-A variant A | healthy runs, phone hot, thermal state serious | app; team lead update 7 Nov |
| First load s | 29 | iPhone 15 | iOS 26.5 | S1-A variant A | healthy runs | app; team lead update 7 Nov |
| Cached load s | 13 | iPhone 15 | iOS 26.5 | S1-A variant A | healthy runs | app; team lead update 7 Nov |
| End to end ms | 930-1380 | iPhone 15 | iOS 26.5 | S1-A variant A | separate failure case: fell back to CPU, not a healthy run | app; team lead update 7 Nov |
| Generator mean ms | 88.3 | iPhone 15 | iOS 26.5 | S1-A variant A | warm, cool phone (thermal nominal), 6 runs | bench.txt 7 Nov 05:45 |
| Generator p95 ms | 91.8 | iPhone 15 | iOS 26.5 | S1-A variant A | warm, cool phone (thermal nominal), 6 runs | bench.txt 7 Nov 05:45 |
| First image ms | 113 | iPhone 15 | iOS 26.5 | S1-A variant A | including safety gate and checks, Best budget | bench.txt 7 Nov 05:45 |
| Cached load s | 15.4 | iPhone 15 | iOS 26.5 | S1-A variant A | models only, one acceleration reload needed | bench.txt 7 Nov 05:45 |

Hosting: no Modal run. The website runs on the Apple M4 through a Cloudflare Quick Tunnel; first image about 1.4 s on the idle M4 with 4 threads.

## 4. Peak memory consumption

| Config | Hardware | Engine | Process max RSS MB (excludes Neural Engine service) | Neural Engine service RSS MB |
|---|---|---|---|---|
| S1-A | Apple M4 | Core ML (coremltools 9.0), CPU_AND_NE | 730.0 | 6.3 |
| S1-FP16 | Apple M4 | Core ML (coremltools 9.0), CPU_AND_NE | 664.0 | 8.6 |
| S0.9-FP16 | Apple M4 | Core ML (coremltools 9.0), CPU_AND_NE | 1048.0 | 11.0 |
| B1 | Apple M4 | PyTorch 2.7 MPS, fp32 compute | 7009.0 | 3.2 |

| iPhone measure | Value | Device | OS | Config | Conditions | Source |
|---|---|---|---|---|---|---|
| Footprint MB | 628-742 | iPhone 15 | iOS 26.5 | S1-A variant A | 12 runs batch | app; team lead report 7 Nov 00:12 |
| Peak MB | about 1198 | iPhone 15 | iOS 26.5 | S1-A variant A | 12 runs batch | app; team lead report 7 Nov 00:12 |
| Peak MB | about 1200 | iPhone 15 | iOS 26.5 | S1-A variant A | healthy runs | app; team lead update 7 Nov |
| Footprint MB | 660-675 | iPhone 15 | iOS 26.5 | S1-A variant A | after load and at end; may omit some Neural Engine memory | bench.txt 7 Nov 05:45 |
| Peak footprint MB | 1212 | iPhone 15 | iOS 26.5 | S1-A variant A | lifetime peak, during load | bench.txt 7 Nov 05:45 |

## 5. Model size

Compiled Core ML, generator only.

| Model | Component | MiB | MB |
|---|---|---|---|
| sdxs-0.9 | taesd_fp16 | 2.4 | 2.5 |
| sdxs-0.9 | text_fp16 | 649.5 | 681.1 |
| sdxs-0.9 | unet_fix_fp16 | 606.1 | 635.6 |
| sdxs-0.9 | **variant FP16 total** | 1258.1 | 1319.2 |
| sdxs-dreamshaper | taesd_fp16 | 2.4 | 2.5 |
| sdxs-dreamshaper | text_fp16 | 234.9 | 246.3 |
| sdxs-dreamshaper | text_pal8 | 153.9 | 161.4 |
| sdxs-dreamshaper | unet_fix_fp16 | 582.4 | 610.6 |
| sdxs-dreamshaper | unet_fix_pal8 | 291.7 | 305.8 |
| sdxs-dreamshaper | **variant A total** | 448.0 | 469.8 |
| sdxs-dreamshaper | **variant FP16 total** | 819.6 | 859.5 |

Adding the checks and safety models on the iPhone: 122.6 MiB more (compiled, measured with du).

| Baseline | MiB | MB |
|---|---|---|
| B0, SD 1.5 fp16 weights | 2033.8 | 2132.6 |
| B1, SD-Turbo fp16 weights | 2460.5 | 2580.1 |


## 6. Quality versus computational cost

| Config | CLIP ViT-L/14 | Mean latency ms | Size MiB | UNet calls | Hardware |
|---|---|---|---|---|---|
| S1-A | 26.637 [26.438, 26.841] | 57.5 | 448.0 | 1 | Apple M4 |
| S1-FP16 | 26.643 [26.44, 26.85] | 63.8 | 819.6 | 1 | Apple M4 |
| S0.9-FP16 | 26.194 [25.999, 26.388] | 84.4 | 1258.1 | 1 | Apple M4 |
| B1 | 26.176 [25.983, 26.369] | 1364.8 | 2460.5 | 1 | Apple M4 |

![pareto_latency.png](results/pareto_latency.png)

![pareto_size.png](results/pareto_size.png)

Search: best of 8 against one image, 654 held-out complex prompts, Apple M4 Core ML, original wording. The runtime score picks; DETR-R101 and CLIP L/14 grade. Percentages except the CLIP row.

| Check | n | One image | Best of 8 | Difference | 95% CI |
|---|---|---|---|---|---|
| Single objects | 61 | 95.08 | 96.72 | 1.64 | [0.0, 4.92] |
| Single colours | 72 | 73.61 | 79.17 | 5.56 | [-2.78, 13.89] |
| Two objects | 76 | 39.47 | 52.63 | 13.16 | [3.95, 22.37] |
| Counts | 61 | 32.79 | 39.34 | 6.56 | [-1.64, 16.39] |
| Spatial | 92 | 23.91 | 27.17 | 3.26 | [-2.17, 8.7] |
| Colours on the right objects | 77 | 9.09 | 6.49 | -2.6 | [-7.79, 2.6] |
| Positions | 77 | 1.3 | 6.49 | 5.19 | [0.0, 11.69] |
| All checkable prompts | 516 | 37.02 | 41.67 | 4.65 | [2.13, 7.36] |
| CLIP L/14 score | 654 | 27.29 | 28.03 | 0.74 | [0.56, 0.93] |

Classical steps on against off: not measured.

Complex prompts, GenEval rules with DETR-R101 boxes. Not comparable with published GenEval numbers.

| Config | Split | Tag | Accuracy | n |
|---|---|---|---|---|
| S1-A | complex_test | color_attr | 0.0909 | 77 |
| S1-A | complex_test | colors | 0.7361 | 72 |
| S1-A | complex_test | counting | 0.3279 | 61 |
| S1-A | complex_test | position | 0.013 | 77 |
| S1-A | complex_test | single_object | 0.9508 | 61 |
| S1-A | complex_test | spatial | 0.2391 | 92 |
| S1-A | complex_test | two_object | 0.3947 | 76 |

Hand labels: not done.
