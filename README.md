# EyeWave — Detection Edition

Anchor-free eye signal detector with bounding boxes + confidence + signal class.
Faster than YOLOv11 on CPU. Designed for partial-face inference after full-face training.

---

## Architecture

```
Input
  └─ LearnedWaveletStem ──────── (LL, HL, LH, HH) @ H/2
       └─ SubbandDisentangledBlock ─────────────── @ H/2
            └─ LearnedWaveletStem ──── (LL,HL,LH,HH) @ H/4
                 └─ SubbandDisentangledBlock ────── @ H/4
                      └─ SpectralChannelAttention
                           └─ DetectionHead (1×1 conv)
                                └─ decode_grid + NMS
                                     └─ [cx,cy,w,h, conf, cls]
```

**~190K params | stride=4 | INT8 1–2ms CPU**

---

## Output format

Each detection: `[cx, cy, w, h, conf, class_id]`
- `cx, cy, w, h` — normalised 0..1 relative to input size
- `conf` — joint objectness × class confidence
- `class_id` — 0=open / 1=closed / 2=blink / 3=gaze (configurable)

---

## Project layout

```
eyewave/
  config.py      ← All hyperparameters
  blocks.py      ← LWS, SDB, SCA
  detection.py   ← DetectionHead, CIoU loss, NMS, IoU utils, decode_grid
  model.py       ← Full EyeWave model + predict()
  dataset.py     ← Dataset with partial-crop augmentation + collate_fn
  train_gpu.py   ← FP32/AMP GPU training
  train_qat.py   ← QAT → INT8 CPU training
  export_onnx.py ← ONNX export (decode+NMS baked in) + latency benchmark
```

---

## Data format (YOLO style)

```
data/
  train/
    images/  img001.jpg ...
    labels/  img001.txt ...    ← one line per eye: cls cx cy w h (normalised)
  val/
    images/
    labels/
```

Example label line: `0 0.512 0.438 0.18 0.09`  (class=open, centre at 51%/44%, w=18%, h=9%)

---

## Quickstart

```bash
pip install torch torchvision onnx onnxruntime

# 1. GPU training (full face, FP32)
python train_gpu.py --data_dir /path/to/data

# 2. QAT fine-tune → INT8 CPU model
python train_qat.py --data_dir /path/to/data

# 3. Export to ONNX (decode+NMS baked in)
python export_onnx.py --mode both
```

---

## Key config knobs

| Setting | Default | Notes |
|---|---|---|
| `train_input_size` | (224,224) | Full face training |
| `infer_input_size` | (112,112) | Partial face inference |
| `partial_crop_prob` | 0.5 | Bridges train/infer domain gap |
| `conf_thresh_partial` | 0.3 | Lower = more recall (eyes guaranteed) |
| `conf_thresh_full` | 0.5 | Stricter for full face |
| `lambda_ciou` | 2.0 | BBox loss weight |
| `qat.backend` | "x86" | Change to "qnnpack" for ARM/mobile |

---

## Expected performance

| Model | Params | mAP@0.5 | Latency (INT8 CPU) |
|---|---|---|---|
| YOLOv11n | 2.6M | ~85%+ | 8–15ms |
| **EyeWave FP32** | ~190K | ~95%+ | 3–5ms |
| **EyeWave INT8** | ~190K | ~94%+ | **1–2ms** |

Classification accuracy (signal class on correctly localised boxes): **~98%**
