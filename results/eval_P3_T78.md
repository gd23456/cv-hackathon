1RUA24CSE0146,1RUA24CSE0201,1RUA24CSE0139,1RUA24CSE0196

## 1. CLIP similarity

Headline ViT-L/14 via torchmetrics, max(100 x cos, 0), against the original prompt, with bootstrap 95% CI. ViT-B/32 is raw cosine.

| Config | Prompts | n | CLIP ViT-L/14 | CLIP ViT-B/32 cosine | Hardware | Engine |
|---|---|---|---|---|---|---|
| S1-A | coco_1k | 1000 | 26.637 [26.438, 26.841] | 0.312 | Apple M4 | Core ML (coremltools 9.0), CPU_AND_NE |
| S1-FP16 | coco_1k | 1000 | 26.643 [26.44, 26.85] | 0.3119 | Apple M4 | Core ML (coremltools 9.0), CPU_AND_NE |
| S0.9-FP16 | coco_1k | 1000 | 26.194 [25.999, 26.388] | 0.3128 | Apple M4 | Core ML (coremltools 9.0), CPU_AND_NE |

Flickr30K, separate protocol: TODO: not measured yet

iPhone 15: TODO: not measured yet (quality is measured on M4 with the same Core ML models; device output not yet compared).

## 2. FID

clean-fid clean mode, PNG, 5,000 generated vs the same 5,000 COCO val2017 references (center crop, 512 Lanczos).

FID at 5k: TODO: not measured yet

KID and CMMD are for ablations; FID is not reported below 5k. CMMD uses ViT-L/14 at 224, the paper uses 336.

| Config | Split | n | KID x 1000 | CMMD x 1000 |
|---|---|---|---|---|
| S1-A | coco_1k | 1000 | 6.93 | 0.769 |
| S1-FP16 | coco_1k | 1000 | 6.972 | 0.765 |
| S0.9-FP16 | coco_1k | 1000 | 8.808 | 0.643 |

## 3. Image generation latency

| Config | Hardware | Engine | Warm n | Mean ms | p50 ms | p95 ms | Text ms | UNet ms | Decoder ms | Cold to first image s |
|---|---|---|---|---|---|---|---|---|---|---|

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

iPhone p50 and p95: TODO: not measured yet (from the app).

Modal: TODO: not measured yet

## 4. Peak memory consumption

| Config | Hardware | Engine | Process max RSS MB (excludes Neural Engine service) | Neural Engine service RSS MB |
|---|---|---|---|---|

| iPhone measure | Value | Device | OS | Config | Conditions | Source |
|---|---|---|---|---|---|---|
| Footprint MB | 628-742 | iPhone 15 | iOS 26.5 | S1-A variant A | 12 runs batch | app; team lead report 7 Nov 00:12 |
| Peak MB | about 1198 | iPhone 15 | iOS 26.5 | S1-A variant A | 12 runs batch | app; team lead report 7 Nov 00:12 |
| Peak MB | about 1200 | iPhone 15 | iOS 26.5 | S1-A variant A | healthy runs | app; team lead update 7 Nov |

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

With verifier on iPhone: TODO: not measured yet

| Baseline | MiB | MB |
|---|---|---|
| B1, SD-Turbo fp16 weights | 2460.5 | 2580.1 |


## 6. Quality versus computational cost

| Config | CLIP ViT-L/14 | Mean latency ms | Size MiB | UNet calls | Hardware |
|---|---|---|---|---|---|
| S1-A | 26.637 [26.438, 26.841] | TODO: not measured yet | TODO: not measured yet | 1 | Apple M4 |
| S1-FP16 | 26.643 [26.44, 26.85] | TODO: not measured yet | TODO: not measured yet | 1 | Apple M4 |
| S0.9-FP16 | 26.194 [25.999, 26.388] | TODO: not measured yet | TODO: not measured yet | 1 | Apple M4 |

pareto_latency.png: TODO: not measured yet

pareto_size.png: TODO: not measured yet

e1.png: TODO: not measured yet

ablation_classical.png: TODO: not measured yet

E1: TODO: not measured yet

Complex prompts: TODO: not measured yet

Hand labels, about 100 images: TODO: not measured yet (team fills results/hand_labels/*.csv).
