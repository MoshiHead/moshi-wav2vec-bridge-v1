"""
compare_inference.py — Side-by-side comparison of real Wav2Vec2 vs Bridge-model output
========================================================================================
Pipeline
--------
  WAV  ──┬──► Wav2Vec2Extractor (HF)    → wav2vec2_gt (T_h, 768)    ← ground-truth @ 25 Hz
         │
         └──► MimiExtractor             → mimi_tokens (T_m, 8)      @ 12.5 Hz
               └──► BridgeInference     → bridge_pred (2*T_m, 768)  ← prediction @ 25 Hz

Outputs (always saved automatically when called via --compare)
--------------------------------------------------------------
  • bridge_pred_features.npy   — Bridge model output  (numpy float32)
  • wav2vec2_gt_features.npy   — Wav2Vec2 HF output   (numpy float32)

Error Metrics (after aligning lengths)
---------------------------------------
  • MSE  (mean squared error)
  • MAE  (mean absolute error)
  • RMSE (root mean squared error)
  • cosine similarity  (mean over frames)
  • SNR  (signal-to-noise ratio, dB)
  • per-dimension RMSE  (top-5 worst dims printed)

Usage
-----
  python compare_inference.py \\
      --audio path/to/audio.wav \\
      --checkpoint checkpoints/best.pt \\
      --config config.yaml

Optional flags
--------------
  --wav2vec2-model      HF repo or local path for Wav2Vec2 (overrides config paths.wav2vec2_model)
  --mimi-model          HF repo or local path for Mimi (overrides config paths.mimi_model)
  --device              cuda | cpu  (default: auto)
  --save-gt             path.pt     save ground-truth Wav2Vec2 features as .pt
  --save-pred           path.pt     save bridge prediction features as .pt
  --save-gt-npy         path.npy    save ground-truth features as .npy (default: wav2vec2_gt_features.npy)
  --save-pred-npy       path.npy    save bridge prediction as .npy (default: bridge_pred_features.npy)
  --no-auto-save-npy    disable automatic .npy saving
  --plot                show matplotlib comparison plots (requires matplotlib)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Error metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(gt: torch.Tensor, pred: torch.Tensor) -> dict:
    """
    Compute comparison metrics between two feature matrices.

    Parameters
    ----------
    gt   : (T, D) float32 — ground-truth HuBERT features
    pred : (T, D) float32 — bridge model prediction (already aligned to gt length)

    Returns
    -------
    dict with float values for each metric.
    """
    assert gt.shape == pred.shape, (
        f"Shape mismatch after alignment: gt={gt.shape}, pred={pred.shape}"
    )
    diff = gt - pred

    mse  = diff.pow(2).mean().item()
    mae  = diff.abs().mean().item()
    rmse = mse ** 0.5

    # Cosine similarity per frame → mean
    cos = torch.nn.functional.cosine_similarity(gt, pred, dim=-1)  # (T,)
    mean_cos = cos.mean().item()

    # SNR: 10*log10(signal_power / noise_power)
    signal_power = gt.pow(2).mean().item()
    noise_power  = diff.pow(2).mean().item()
    snr_db = 10.0 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12))

    # Per-dimension RMSE  → (D,)
    per_dim_rmse = diff.pow(2).mean(dim=0).sqrt()  # (D,)

    return {
        "mse":          mse,
        "mae":          mae,
        "rmse":         rmse,
        "mean_cosine":  mean_cos,
        "snr_db":       snr_db,
        "per_dim_rmse": per_dim_rmse,   # kept as tensor for downstream use
    }


def _quality_label(cos: float, snr: float) -> str:
    """Return a human-readable quality verdict based on cosine sim and SNR."""
    if cos >= 0.95 and snr >= 20:
        return "🟢  EXCELLENT"
    elif cos >= 0.85 and snr >= 10:
        return "🟡  GOOD"
    elif cos >= 0.70 and snr >= 5:
        return "🟠  FAIR"
    else:
        return "🔴  POOR"


def print_metrics(metrics: dict, gt_shape: tuple, pred_shape: tuple,
                  saved_files: Optional[list] = None):
    """Pretty-print the comparison metrics with quality verdict."""
    bar  = "═" * 66
    thin = "─" * 66

    print(f"\n{bar}")
    print("   Wav2Vec2 Ground-Truth  ◄vs►  Bridge Model Prediction")
    print(bar)
    print(f"  Ground-truth shape  : {gt_shape}")
    print(f"  Prediction shape    : {pred_shape}")
    print(thin)
    print(f"  MSE             : {metrics['mse']:.6f}")
    print(f"  MAE             : {metrics['mae']:.6f}")
    print(f"  RMSE            : {metrics['rmse']:.6f}")
    print(f"  Mean cos-sim    : {metrics['mean_cosine']:.6f}  (1.0 = perfect)")
    print(f"  SNR             : {metrics['snr_db']:.2f} dB   (higher = better)")
    print(thin)

    verdict = _quality_label(metrics["mean_cosine"], metrics["snr_db"])
    print(f"  Overall quality : {verdict}")
    print(thin)

    # Top-5 worst dimensions
    per_dim = metrics["per_dim_rmse"]
    top5_vals, top5_idx = torch.topk(per_dim, min(5, len(per_dim)))
    print("  Top-5 worst dimensions (by per-dim RMSE):")
    for rank, (idx, val) in enumerate(zip(top5_idx.tolist(), top5_vals.tolist()), 1):
        bar_len = int(val / (top5_vals[0].item() + 1e-9) * 20)
        bar_str = "█" * bar_len
        print(f"    #{rank}  dim={idx:4d}   RMSE={val:.6f}  {bar_str}")

    if saved_files:
        print(thin)
        print("  Saved outputs:")
        for label, fpath in saved_files:
            print(f"    {label:<28s} → {fpath}")

    print(bar)


# ──────────────────────────────────────────────────────────────────────────────
# Feature alignment helper
# ──────────────────────────────────────────────────────────────────────────────

def align_frames(gt: torch.Tensor, pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Trim both tensors to their common minimum length along dimension 0.
    Both HuBERT GT (ONNX, 25 Hz) and bridge prediction (Mimi × upsample_factor 2 = 25 Hz)
    are at the same rate, so no upsampling/downsampling is required — only a trim to
    handle minor off-by-one rounding differences.
    """
    T = min(gt.shape[0], pred.shape[0])
    return gt[:T], pred[:T]


# ──────────────────────────────────────────────────────────────────────────────
# Main comparison function
# ──────────────────────────────────────────────────────────────────────────────

def compare(
    audio_path:              str,
    checkpoint_path:         str,
    config_path:             str,
    device:                  Optional[str] = None,
    wav2vec2_model_override: Optional[str] = None,
    mimi_model_override:     Optional[str] = None,
    save_gt:                 Optional[str] = None,
    save_pred:               Optional[str] = None,
    save_gt_npy:             Optional[str] = "wav2vec2_gt_features.npy",
    save_pred_npy:           Optional[str] = "bridge_pred_features.npy",
    auto_save_npy:           bool = True,
    plot:                    bool = False,
):
    """
    Full pipeline: WAV → Wav2Vec2_gt + Bridge_pred → error metrics.

    By default (auto_save_npy=True), both outputs are always saved as .npy files:
      • bridge_pred_features.npy
      • wav2vec2_gt_features.npy
    """
    # ── Load config ───────────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ── Resolve device ────────────────────────────────────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    logger.info(f"Using device: {dev}")

    # ── Override model paths from CLI if provided ─────────────────────────────
    wav2vec2_model = wav2vec2_model_override or cfg["paths"]["wav2vec2_model"]
    mimi_model_name = mimi_model_override    or cfg["paths"]["mimi_model"]

    # ── Load audio ────────────────────────────────────────────────────────────
    import torchaudio
    logger.info(f"Loading audio: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    # Keep original SR — each extractor resamples internally

    duration_s = waveform.shape[-1] / sr
    logger.info(f"Audio: {duration_s:.2f}s @ {sr} Hz, shape={tuple(waveform.shape)}")

    # ─────────────────────────────────────────────────────────────────────────
    # BRANCH 1: Ground-truth features via facebook/wav2vec2-base-960h
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/3] Extracting ground-truth Wav2Vec2 features (HuggingFace)…")
    from dataset import Wav2Vec2Extractor
    wav2vec2_extractor = Wav2Vec2Extractor(
        model_name=wav2vec2_model,
        device=device,
    )
    wav2vec2_gt = wav2vec2_extractor.extract(waveform, sr)  # (T_h, 768)  25 Hz
    print(f"      Wav2Vec2 GT shape: {tuple(wav2vec2_gt.shape)}  @ ~25 Hz")

    # ─────────────────────────────────────────────────────────────────────────
    # BRANCH 2: Mimi tokens → Bridge model
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/3] Extracting Mimi tokens + running Bridge model…")
    from dataset import MimiExtractor
    from inference import BridgeInference

    mimi_extractor = MimiExtractor(mimi_model_name, device=device)
    mimi_tokens = mimi_extractor.extract(waveform, sr)   # (T_m, num_codebooks)
    print(f"      Mimi tokens shape: {tuple(mimi_tokens.shape)}  @ 12.5 Hz")

    bridge = BridgeInference(checkpoint_path, config_path, device=device)
    bridge_pred = bridge(mimi_tokens)                    # (1, 2*T_m, 768) → cpu
    bridge_pred = bridge_pred.squeeze(0)                 # (2*T_m, 768)     25 Hz
    print(f"      Bridge pred shape: {tuple(bridge_pred.shape)}  @ 25 Hz")

    # ─────────────────────────────────────────────────────────────────────────
    # Align lengths: both Wav2Vec2 GT and bridge pred are at 25 Hz.
    # Just trim to the shorter of the two (handles rounding off-by-one).
    # ─────────────────────────────────────────────────────────────────────────
    gt_aligned, pred_aligned = align_frames(wav2vec2_gt, bridge_pred)
    rate_label = "25 Hz"

    print(f"\n      Aligned at {rate_label}: gt={tuple(gt_aligned.shape)}, "
          f"pred={tuple(pred_aligned.shape)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Compute metrics
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/3] Computing error metrics…")
    metrics = compute_metrics(
        gt_aligned.float(),
        pred_aligned.float(),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Save outputs — .npy (automatic) and .pt (optional)
    # ─────────────────────────────────────────────────────────────────────────
    saved_files = []

    # Always save .npy unless explicitly disabled
    if auto_save_npy:
        _pred_npy = save_pred_npy or "bridge_pred_features.npy"
        _gt_npy   = save_gt_npy   or "wav2vec2_gt_features.npy"

        np.save(_pred_npy, pred_aligned.float().numpy())
        saved_files.append(("Bridge pred (npy)", _pred_npy))

        np.save(_gt_npy, gt_aligned.float().numpy())
        saved_files.append(("Wav2Vec2 GT (npy)", _gt_npy))

    # Optional .pt saves (legacy / interop)
    if save_gt:
        torch.save(gt_aligned, save_gt)
        saved_files.append(("Wav2Vec2 GT (pt) ", save_gt))
    if save_pred:
        torch.save(pred_aligned, save_pred)
        saved_files.append(("Bridge pred (pt) ", save_pred))

    # ─────────────────────────────────────────────────────────────────────────
    # Display metrics
    # ─────────────────────────────────────────────────────────────────────────
    print_metrics(
        metrics,
        gt_shape=tuple(gt_aligned.shape),
        pred_shape=tuple(pred_aligned.shape),
        saved_files=saved_files if saved_files else None,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Optional: matplotlib visualisation
    # ─────────────────────────────────────────────────────────────────────────
    if plot:
        _plot_comparison(gt_aligned, pred_aligned, metrics, rate_label)

    return metrics, gt_aligned, pred_aligned


# ──────────────────────────────────────────────────────────────────────────────
# Plotting (optional)
# ──────────────────────────────────────────────────────────────────────────────

def _plot_comparison(
    gt: torch.Tensor,
    pred: torch.Tensor,
    metrics: dict,
    rate_label: str,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot. pip install matplotlib")
        return

    gt_np   = gt.numpy()
    pred_np = pred.numpy()
    diff_np = (gt - pred).abs().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    fig.suptitle(
        f"HuBERT GT vs Bridge Prediction @ {rate_label}\n"
        f"RMSE={metrics['rmse']:.4f}  COS={metrics['mean_cosine']:.4f}  "
        f"SNR={metrics['snr_db']:.1f} dB",
        fontsize=12, fontweight="bold",
    )

    # ── Heatmap: GT features ─────────────────────────────────────────────────
    ax = axes[0]
    im0 = ax.imshow(
        gt_np.T, aspect="auto", origin="lower",
        vmin=np.percentile(gt_np, 5), vmax=np.percentile(gt_np, 95),
        cmap="magma",
    )
    ax.set_title("Wav2Vec2 Ground-Truth (facebook/wav2vec2-base-960h)")
    ax.set_ylabel("Dimension")
    plt.colorbar(im0, ax=ax, fraction=0.015, pad=0.01)

    # ── Heatmap: Bridge prediction ────────────────────────────────────────────
    ax = axes[1]
    im1 = ax.imshow(
        pred_np.T, aspect="auto", origin="lower",
        vmin=np.percentile(gt_np, 5), vmax=np.percentile(gt_np, 95),
        cmap="magma",
    )
    ax.set_title("Bridge Model Prediction")
    ax.set_ylabel("Dimension")
    plt.colorbar(im1, ax=ax, fraction=0.015, pad=0.01)

    # ── Heatmap: Absolute error ───────────────────────────────────────────────
    ax = axes[2]
    im2 = ax.imshow(
        diff_np.T, aspect="auto", origin="lower",
        cmap="hot",
    )
    ax.set_title("Absolute Error |GT - Pred|")
    ax.set_xlabel(f"Frame (@ {rate_label})")
    ax.set_ylabel("Dimension")
    plt.colorbar(im2, ax=ax, fraction=0.015, pad=0.01)

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare real HuBERT features vs Bridge model output for a WAV file."
    )
    parser.add_argument("--audio",       required=True,
                        help="Input WAV/FLAC audio file")
    parser.add_argument("--checkpoint",  required=True,
                        help="Trained bridge model checkpoint (.pt)")
    parser.add_argument("--config",      required=True,
                        help="config.yaml path")
    parser.add_argument("--device",      default=None,
                        help="Force device: cuda | cpu (default: auto-detect)")
    parser.add_argument("--wav2vec2-model", default=None,
                        help="Override HF repo or local path for Wav2Vec2 "
                             "(default: paths.wav2vec2_model from config.yaml)")
    parser.add_argument("--mimi-model",  default=None,
                        help="Override Mimi HF repo or local path "
                             "(default: paths.mimi_model from config.yaml)")
    # .pt saves (optional, legacy)
    parser.add_argument("--save-gt",     default=None,
                        help="Save ground-truth Wav2Vec2 features to this .pt file")
    parser.add_argument("--save-pred",   default=None,
                        help="Save bridge prediction features to this .pt file")
    # .npy saves (automatic by default)
    parser.add_argument("--save-gt-npy",   default="wav2vec2_gt_features.npy",
                        help="Path for ground-truth .npy output "
                             "(default: wav2vec2_gt_features.npy)")
    parser.add_argument("--save-pred-npy", default="bridge_pred_features.npy",
                        help="Path for bridge prediction .npy output "
                             "(default: bridge_pred_features.npy)")
    parser.add_argument("--no-auto-save-npy", action="store_true",
                        help="Disable automatic .npy saving")
    parser.add_argument("--plot",        action="store_true",
                        help="Show matplotlib heatmap comparison (requires matplotlib)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    compare(
        audio_path              = args.audio,
        checkpoint_path         = args.checkpoint,
        config_path             = args.config,
        device                  = args.device,
        wav2vec2_model_override = args.wav2vec2_model,
        mimi_model_override     = args.mimi_model,
        save_gt                 = args.save_gt,
        save_pred               = args.save_pred,
        save_gt_npy             = args.save_gt_npy,
        save_pred_npy           = args.save_pred_npy,
        auto_save_npy           = not args.no_auto_save_npy,
        plot                    = args.plot,
    )


if __name__ == "__main__":
    main()
