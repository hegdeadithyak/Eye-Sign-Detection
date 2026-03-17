"""
EyeWave — Intel RealSense D435 / D345 Live Prediction
-------------------------------------------------------
Uses pyrealsense2 to grab aligned RGB + Depth frames.
For each detected bbox the script reads the real-world
distance (metres) from the depth frame and overlays it.

Requirements:
    pip install pyrealsense2 opencv-python numpy torch

Usage (from ~/eyewave):
    python live_realsense.py
    python live_realsense.py --conf 0.05 --class_names Normal Abnormal
    python live_realsense.py --save out.mp4
    python live_realsense.py --depth_view   # side-by-side colour+depth

Controls:
    Q / ESC  → quit
    + / =    → raise conf threshold +0.05
    -        → lower conf threshold -0.05
    D        → toggle depth overlay
    C        → toggle depth colourmap panel
    S        → screenshot
    P        → pause / resume
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import pyrealsense2 as rs
except ImportError:
    print("❌  pyrealsense2 not found.")
    print("    Install with:  pip install pyrealsense2")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CFG, EyeWaveConfig
from model import build_model
from detection import decode_grid, nms


# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = [
    (  0, 220,  90),
    ( 30, 144, 255),
    (255, 165,   0),
    (220,  20,  60),
    (148,   0, 211),
    (  0, 206, 209),
    (255, 215,   0),
    (255,  20, 147),
]

def colour_for(cls_idx: int):
    return PALETTE[cls_idx % len(PALETTE)]


# ── Pre-processing ─────────────────────────────────────────────────────────────

def preprocess(frame_bgr: np.ndarray, input_hw) -> torch.Tensor:
    """BGR uint8 HWC → grayscale normalised tensor (1, 1, H, W).
    Model stem weight [64, 1, 2, 2] requires exactly 1 input channel."""
    h, w = input_hw
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)     # (H, W) uint8
    gray = cv2.resize(gray, (w, h)).astype(np.float32) / 255.0
    gray = (gray - 0.5) / 0.5                              # normalise → [-1, 1]
    return torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


# ── RealSense pipeline ─────────────────────────────────────────────────────────

def make_pipeline(color_w=640, color_h=480, depth_w=640, depth_h=480, fps=30):
    """
    Configure and start a RealSense pipeline.
    Returns (pipeline, align_to_color, depth_scale).
    """
    pipeline = rs.pipeline()
    config   = rs.config()

    # Check which streams the connected device supports
    ctx        = rs.context()
    devices    = ctx.query_devices()
    if len(devices) == 0:
        print("❌  No RealSense device found. Is the camera plugged in?")
        sys.exit(1)

    dev  = devices[0]
    name = dev.get_info(rs.camera_info.name)
    sn   = dev.get_info(rs.camera_info.serial_number)
    fw   = dev.get_info(rs.camera_info.firmware_version)
    print(f"[RS] Device  : {name}")
    print(f"[RS] Serial  : {sn}")
    print(f"[RS] Firmware: {fw}")

    config.enable_stream(rs.stream.color, color_w, color_h,
                         rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, depth_w, depth_h,
                         rs.format.z16,  fps)

    profile = pipeline.start(config)

    # Depth scale: converts raw uint16 → metres
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()
    print(f"[RS] Depth scale: {depth_scale:.5f} m/unit")

    # Align depth to colour frame
    align = rs.align(rs.stream.color)

    # Auto-exposure warm-up
    print("[RS] Auto-exposure warm-up (30 frames) …")
    for _ in range(30):
        pipeline.wait_for_frames()

    return pipeline, align, depth_scale


def get_depth_at_box(depth_frame_np: np.ndarray,
                     x1: int, y1: int, x2: int, y2: int,
                     depth_scale: float,
                     method: str = "median") -> float:
    """
    Return distance (metres) in the bbox region.
    Uses a 5×5 patch around the bbox centre to avoid edge noise.
    Zero depth values (invalid) are excluded.
    """
    h, w = depth_frame_np.shape
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    r  = 5   # half-patch radius
    patch = depth_frame_np[
        max(0, cy - r): min(h, cy + r + 1),
        max(0, cx - r): min(w, cx + r + 1),
    ]
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0.0
    raw = float(np.median(valid)) if method == "median" else float(np.mean(valid))
    return raw * depth_scale


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_predictions(frame: np.ndarray,
                     dets: torch.Tensor,
                     class_names: list,
                     depth_frame_np: np.ndarray,
                     depth_scale: float,
                     show_depth: bool):
    oh, ow = frame.shape[:2]

    for d in range(dets.shape[0]):
        cx, cy, bw, bh, conf, cls = dets[d, :6].tolist()
        x1 = int((cx - bw / 2) * ow);  y1 = int((cy - bh / 2) * oh)
        x2 = int((cx + bw / 2) * ow);  y2 = int((cy + bh / 2) * oh)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(ow - 1, x2), min(oh - 1, y2)

        cls_int  = int(cls)
        cls_name = class_names[cls_int] if cls_int < len(class_names) else f"cls_{cls_int}"
        color    = colour_for(cls_int)

        # Main bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Depth reading
        dist_str = ""
        if show_depth and depth_frame_np is not None:
            dist_m = get_depth_at_box(depth_frame_np, x1, y1, x2, y2, depth_scale)
            dist_str = f"  {dist_m:.2f}m" if dist_m > 0 else "  --m"

        label = f"{cls_name} {conf:.2f}{dist_str}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        ly = max(y1, th + 6)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 6, ly + bl - 2), color, -1)
        cv2.putText(frame, label, (x1 + 3, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

        # Depth: crosshair at bbox centre
        if show_depth:
            bmx, bmy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.drawMarker(frame, (bmx, bmy), color,
                           cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)

    return frame


def draw_hud(frame: np.ndarray, fps: float, n_dets: int,
             conf_thresh: float, show_depth: bool,
             show_cmap: bool, paused: bool):
    h, w = frame.shape[:2]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"FPS: {fps:5.1f}", ( 8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (  0, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Dets: {n_dets}",  (155, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 200,   0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Conf: {conf_thresh:.3f} [+/-]", (260, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (100, 180, 255), 1, cv2.LINE_AA)
    depth_col = (  0, 255, 200) if show_depth else (120, 120, 120)
    cv2.putText(frame, f"Depth[D]:{'ON' if show_depth else 'OFF'}", (450, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, depth_col, 1, cv2.LINE_AA)
    cmap_col  = (  0, 200, 255) if show_cmap else (120, 120, 120)
    cv2.putText(frame, f"CMap[C]:{'ON' if show_cmap else 'OFF'}", (570, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, cmap_col, 1, cv2.LINE_AA)

    if paused:
        cv2.putText(frame, "PAUSED", (w // 2 - 60, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 80, 255), 2, cv2.LINE_AA)

    # Bottom hint
    hint = "Q/ESC:quit  S:screenshot  P:pause  D:depth  C:cmap  +/-:conf"
    cv2.putText(frame, hint, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (130, 130, 130), 1, cv2.LINE_AA)

    return frame


def depth_colormap(depth_frame_np: np.ndarray) -> np.ndarray:
    """Render the raw depth array as a JET colourmap image."""
    depth_8u = cv2.convertScaleAbs(depth_frame_np, alpha=0.03)
    return cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)


# ── Main ───────────────────────────────────────────────────────────────────────

def run(ckpt_path: str,
        conf_override: float,
        class_names: list,
        save_path: str,
        color_res: tuple,
        fps_target: int,
        show_cmap_default: bool):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Live] Device : {device}")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    print(f"[Live] Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: EyeWaveConfig = ckpt.get("cfg", CFG)

    conf_thresh = conf_override if conf_override is not None else cfg.infer.conf_thresh_full
    print(f"[Live] Checkpoint epoch={ckpt.get('epoch')}  "
          f"mAP={ckpt.get('map50', 0):.4f}  conf={conf_thresh:.3f}")

    model = build_model(cfg.model, cfg.infer).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Live] Model: {model.count_parameters():,} parameters")

    input_hw = cfg.model.train_input_size

    if class_names is None:
        class_names = [f"cls_{i}" for i in range(cfg.model.num_classes)]

    # ── RealSense ─────────────────────────────────────────────────────────────
    cw, ch = color_res
    pipeline, align, depth_scale = make_pipeline(
        color_w=cw, color_h=ch,
        depth_w=cw, depth_h=ch,
        fps=fps_target,
    )

    # ── Warm-up inference ─────────────────────────────────────────────────────
    with torch.no_grad():
        _ = model(torch.zeros(1, 1, *input_hw, device=device))  # 1-ch grayscale
    print("[Live] Ready. Controls: Q=quit  D=depth  C=cmap  S=screenshot\n")

    # ── Video writer ──────────────────────────────────────────────────────────
    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps_target, (cw, ch))
        print(f"[Live] Saving → {save_path}")

    fps_buf      = deque(maxlen=30)
    show_depth   = True
    show_cmap    = show_cmap_default
    paused       = False
    screenshot_n = 0
    frame_count  = 0
    last_display = None

    cv2.namedWindow("EyeWave — RealSense", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("EyeWave — RealSense", cw, ch)

    try:
        while True:
            t0  = time.perf_counter()
            key = cv2.waitKey(1) & 0xFF

            # ── Keys ──────────────────────────────────────────────────────────
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('+'), ord('=')):
                conf_thresh = min(conf_thresh + 0.05, 0.99)
                print(f"  conf → {conf_thresh:.3f}")
            elif key == ord('-'):
                conf_thresh = max(conf_thresh - 0.05, 0.001)
                print(f"  conf → {conf_thresh:.3f}")
            elif key in (ord('d'), ord('D')):
                show_depth = not show_depth
                print(f"  Depth overlay: {'ON' if show_depth else 'OFF'}")
            elif key in (ord('c'), ord('C')):
                show_cmap = not show_cmap
                print(f"  Depth colourmap: {'ON' if show_cmap else 'OFF'}")
            elif key in (ord('p'), ord('P')):
                paused = not paused
                print(f"  {'Paused' if paused else 'Resumed'}")
            elif key in (ord('s'), ord('S')):
                if last_display is not None:
                    fn = f"rs_screenshot_{screenshot_n:04d}.png"
                    cv2.imwrite(fn, last_display)
                    print(f"  Screenshot → {fn}")
                    screenshot_n += 1

            if paused:
                if last_display is not None:
                    cv2.imshow("EyeWave — RealSense", last_display)
                continue

            # ── Grab aligned frames ───────────────────────────────────────────
            try:
                frames        = pipeline.wait_for_frames(timeout_ms=5000)
                aligned       = align.process(frames)
                color_frame   = aligned.get_color_frame()
                depth_frame_rs = aligned.get_depth_frame()
            except RuntimeError as e:
                print(f"[RS] Frame timeout: {e}")
                continue

            if not color_frame or not depth_frame_rs:
                continue

            frame_count += 1

            # NumPy arrays
            color_np = np.asanyarray(color_frame.get_data())   # BGR, HxWx3
            depth_np = np.asanyarray(depth_frame_rs.get_data()) # uint16, HxW (raw)

            # ── Inference ─────────────────────────────────────────────────────
            with torch.no_grad():
                tensor   = preprocess(color_np, input_hw).to(device)  # BGR→gray inside
                raw_grid = model(tensor)
                decoded  = decode_grid(raw_grid, input_hw, cfg.model.stride)
                dets     = nms(
                    decoded[0],
                    conf_thresh=conf_thresh,
                    iou_thresh=cfg.infer.nms_iou_thresh,
                    max_detections=cfg.model.max_detections,
                )

            # ── Draw on colour frame ───────────────────────────────────────────
            display = draw_predictions(
                color_np.copy(), dets, class_names,
                depth_np, depth_scale, show_depth,
            )

            # Optional: side panel with depth colourmap
            if show_cmap:
                cmap   = depth_colormap(depth_np)
                # Resize to match colour frame if needed
                cmap   = cv2.resize(cmap, (cw // 3, ch))
                # Thin separator line
                sep    = np.full((ch, 3, 3), 40, dtype=np.uint8)
                display = np.hstack([display, sep, cmap])
                disp_w = display.shape[1]
            else:
                disp_w = cw

            # FPS
            elapsed = time.perf_counter() - t0
            fps_buf.append(1.0 / max(elapsed, 1e-6))
            fps = sum(fps_buf) / len(fps_buf)

            draw_hud(display, fps, dets.shape[0], conf_thresh,
                     show_depth, show_cmap, paused)

            last_display = display
            cv2.imshow("EyeWave — RealSense", display)

            # Only write colour-only frame to file (not the side-panel)
            if writer:
                writer.write(color_np if show_cmap else display[:ch, :cw])

    except KeyboardInterrupt:
        print("\n[Live] Interrupted.")

    finally:
        pipeline.stop()
        if writer:
            writer.release()
            print(f"[Live] Saved → {save_path}")
        cv2.destroyAllWindows()
        print(f"[Live] Done — {frame_count} frames processed.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",          default="checkpoints/best_gpu.pth")
    parser.add_argument("--conf",          type=float, default=None,
                        help="NMS confidence threshold (e.g. 0.05)")
    parser.add_argument("--class_names",   nargs="+",  default=None,
                        help="Class names in index order")
    parser.add_argument("--save",          default=None,
                        help="Save annotated video to this .mp4 file")
    parser.add_argument("--res",           default="640x480",
                        help="Camera resolution WxH (default 640x480). "
                             "D435 also supports 1280x720, 848x480")
    parser.add_argument("--fps",           type=int,   default=30,
                        help="Camera FPS target (6/15/30/60 — check your device)")
    parser.add_argument("--depth_view",    action="store_true",
                        help="Start with depth colourmap panel enabled")
    args = parser.parse_args()

    try:
        res_w, res_h = map(int, args.res.lower().split("x"))
    except ValueError:
        print(f"❌  Invalid --res format '{args.res}'. Use e.g. 640x480")
        sys.exit(1)

    run(
        ckpt_path       = args.ckpt,
        conf_override   = args.conf,
        class_names     = args.class_names,
        save_path       = args.save,
        color_res       = (res_w, res_h),
        fps_target      = args.fps,
        show_cmap_default = args.depth_view,
    )
