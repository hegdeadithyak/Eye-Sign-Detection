"""
EyeWave Configuration
All hyperparameters in one place — change here, affects everything.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    # ── Input ─────────────────────────────────────────────────────────────────
    input_channels: int = 1
    train_input_size: Tuple[int, int] = (224, 224)  # Full face during training
    infer_input_size: Tuple[int, int] = (112, 112)  # Partial face at inference

    # ── Architecture ──────────────────────────────────────────────────────────
    base_channels: int = 64      # Channels per subband after stage 1
    num_classes: int = 5        # open / closed / blink / gaze

    # ── Detection ─────────────────────────────────────────────────────────────
    # Two DWT stages each halve H and W → total stride = 4
    stride: int = 4
    max_detections: int = 2      # Max eyes per partial crop at inference


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    seed: int = 42
    lr_min: float = 1e-6

    # Loss weights
    lambda_ciou: float = 2.0     # BBox CIoU regression
    lambda_conf: float = 4.0     # Focal loss (increased to force detection)
    lambda_cls: float = 1.0      # Signal class CE (increased)
    lambda_attn: float = 5.0     # Attention supervision loss (forces eye focus)

    # IoU threshold to assign a grid cell as positive
    pos_iou_thresh: float = 0.5

    # Probability of applying partial-face crop augmentation
    # Forces model to learn without full-face context → closes train/infer gap
    partial_crop_prob: float = 0.5

    # Top-crop: keep only the top N% of rows (eye-region focus)
    # 0.6 = keep top 60% → drops mouth/chin.
    # Set to 0.0 to disable.
    top_crop_ratio: float = 0.6

    # Eye-region crop: crop to GT bbox region so model only sees eyes
    use_eye_crop: bool = True            # Enable eye-region cropping
    eye_crop_padding: float = 0.3        # Fractional padding around bbox (0.3 = 30%)

    save_dir: str = "checkpoints"
    save_every: int = 10


@dataclass
class QATConfig:
    pretrained_ckpt: str = "checkpoints/best_gpu.pth"
    epochs: int = 10
    lr: float = 1e-5
    batch_size: int = 32
    freeze_observer_epoch: int = 7   # Stop updating scale/zero-point after this
    freeze_bn_epoch: int = 5         # Freeze BN stats after this
    backend: str = "x86"             # "qnnpack" for ARM/mobile
    save_dir: str = "checkpoints"


@dataclass
class InferenceConfig:
    conf_thresh_full: float = 0.1    # Show more detections during training
    conf_thresh_partial: float = 0.3 
    nms_iou_thresh: float = 0.45
    partial_face_mode: bool = True 


@dataclass
class ONNXConfig:
    opset: int = 17
    input_name: str = "eye_patch"
    output_name: str = "detections"  # Raw grid: [B, H*W, 5+num_classes]
    dynamic_batch: bool = True


@dataclass
class EyeWaveConfig:
    model: ModelConfig    = field(default_factory=ModelConfig)
    train: TrainConfig    = field(default_factory=TrainConfig)
    qat: QATConfig        = field(default_factory=QATConfig)
    infer: InferenceConfig= field(default_factory=InferenceConfig)
    onnx: ONNXConfig      = field(default_factory=ONNXConfig)


CFG = EyeWaveConfig()
