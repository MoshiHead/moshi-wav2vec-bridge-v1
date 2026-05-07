"""
inference.py — Streaming & Batch Inference
==========================================
Provides:
  - BridgeInference: batch mode
  - StreamingBridgeInference: causal chunk-by-chunk streaming
  - CLI entry point  (including --compare mode)

Both modes produce HuBERT-compatible features (50 Hz, output_dim-dim) from
Mimi tokens. output_dim is read from config.yaml (default: 1024 for HuBERT-large).

--compare mode (shortcut to compare_inference.py)
-------------------------------------------------
  python inference.py \\
      --audio  path/to/audio.wav \\
      --checkpoint checkpoints/best.pt \\
      --config config.yaml \\
      --compare

This extracts:
  1. Real HuBERT features via the ONNX model (ground-truth)  → hubert_gt_features.npy
  2. Bridge-model prediction via Mimi tokens                 → bridge_pred_features.npy
Then prints MSE / MAE / RMSE / cosine-similarity / SNR error metrics.

Override default .npy output paths with:
  --save-gt-npy   my_hubert.npy
  --save-pred-npy my_bridge.npy

Disable auto .npy saving:
  --no-auto-save-npy
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml

from model import MimiHuBERTBridge

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Shared checkpoint loader
# ──────────────────────────────────────────────────────────────────────────────

def _load_checkpoint(path: str, model: MimiHuBERTBridge, device: torch.device):
    """
    Load bridge weights from a checkpoint file.
    Supports both full trainer checkpoints ({"bridge": state_dict, ...})
    and bare state_dicts saved directly.
    Uses weights_only=True to avoid arbitrary-code-execution warnings in
    PyTorch >= 2.0 when loading untrusted files.
    """
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # weights_only not supported on older PyTorch — fall back gracefully
        ckpt = torch.load(path, map_location=device)

    sd = ckpt.get("bridge", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"Missing keys in checkpoint: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected}")


# ──────────────────────────────────────────────────────────────────────────────
# Batch Inference
# ──────────────────────────────────────────────────────────────────────────────

class BridgeInference:
    """
    Simple batch inference wrapper.
    Loads a trained bridge checkpoint and converts Mimi tokens → HuBERT-like features.
    """

    def __init__(self, checkpoint_path: str, config_path: str, device: Optional[str] = None):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        # Resolve device: explicit arg > config > auto-detect
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device(self.cfg["inference"].get("device", "cuda"))
        else:
            self.device = torch.device("cpu")

        # output_dim is read from config so it works for both 768 and 1024
        self.output_dim = self.cfg["model"]["output_dim"]
        self.num_codebooks = self.cfg["model"]["num_codebooks"]

        self.model = MimiHuBERTBridge(self.cfg).to(self.device)
        _load_checkpoint(checkpoint_path, self.model, self.device)
        self.model.eval()
        logger.info(
            f"Loaded bridge from {checkpoint_path} on {self.device} "
            f"(output_dim={self.output_dim})"
        )

    @torch.no_grad()
    def __call__(
        self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        tokens : (B, T, num_codebooks) or (T, num_codebooks)  int64
        returns: (B, 2T, output_dim)  float32  — always on CPU
        """
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)

        tokens = tokens.to(self.device)
        features, _ = self.model(tokens)           # (B, 2T, output_dim)  — may be fp16

        # Always return float32 to the caller
        features = features.float()

        if mask is not None:
            # Zero-out padded positions: mask is (B, T_m), expand to (B, 2T, 1)
            mask2 = mask.repeat_interleave(2, dim=-1).to(self.device)   # (B, 2T)
            features = features * mask2.unsqueeze(-1)

        return features.cpu()

    @torch.no_grad()
    def from_audio(self, audio_path: str) -> torch.Tensor:
        """
        End-to-end: audio file → HuBERT-like features.
        Returns: (T_h, output_dim) float32 tensor

        Audio is loaded at its native sample rate and passed as-is to
        MimiExtractor.  The extractor always resamples to 24,000 Hz internally,
        so no pre-resampling is required here.
        """
        from dataset import MimiExtractor
        import torchaudio

        waveform, native_sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        # Do NOT resample here — MimiExtractor handles 24 kHz resampling internally.

        extractor = MimiExtractor(self.cfg["paths"]["mimi_model"])
        tokens = extractor.extract(waveform, native_sr)   # (T, num_codebooks)

        features = self(tokens)                           # (1, 2T, output_dim)
        return features.squeeze(0)                        # (2T, output_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Streaming Inference
# ──────────────────────────────────────────────────────────────────────────────

class StreamingBridgeInference:
    """
    Causal streaming inference with KV-cache.

    Usage:
        stream = StreamingBridgeInference(checkpoint, config)
        stream.reset()
        for mimi_chunk in token_stream:           # (chunk_size, num_codebooks)
            feat_chunk = stream.step(mimi_chunk)  # (2*chunk_size, output_dim)
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        chunk_size: Optional[int] = None,
        device: Optional[str] = None,
    ):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.chunk_size    = chunk_size or self.cfg["inference"].get("chunk_size", 50)
        self.output_dim    = self.cfg["model"]["output_dim"]
        self.num_codebooks = self.cfg["model"]["num_codebooks"]

        self.model = MimiHuBERTBridge(self.cfg).to(self.device)
        _load_checkpoint(checkpoint_path, self.model, self.device)
        self.model.eval()

        self._past_kvs: Optional[list] = None
        self._step_count = 0

    def reset(self):
        """Reset streaming state (call at the start of each new utterance)."""
        self._past_kvs  = None
        self._step_count = 0

    @torch.no_grad()
    def step(self, tokens_chunk: torch.Tensor) -> torch.Tensor:
        """
        tokens_chunk : (C, num_codebooks) or (1, C, num_codebooks)  int64
        returns      : (2C, output_dim)  float32  — on CPU
        """
        if tokens_chunk.dim() == 2:
            tokens_chunk = tokens_chunk.unsqueeze(0)   # (1, C, num_codebooks)

        tokens_chunk = tokens_chunk.to(self.device)

        features, present_kvs = self.model(
            tokens_chunk,
            use_cache=True,
            past_kvs=self._past_kvs,
        )
        self._past_kvs = present_kvs

        # The KV sequence length is in the upsampled (×2) space.
        max_kv_len = self.cfg["model"].get("max_seq_len", 2048) * 2
        if self._past_kvs is not None:
            trimmed = []
            for layer_kv in self._past_kvs:
                if layer_kv is not None:
                    k, v = layer_kv
                    if k.shape[2] > max_kv_len:
                        k = k[:, :, -max_kv_len:]
                        v = v[:, :, -max_kv_len:]
                    trimmed.append((k, v))
                else:
                    trimmed.append(None)
            self._past_kvs = trimmed

        self._step_count += 1
        return features.squeeze(0).float().cpu()   # (2C, output_dim)

    def stream_tokens(self, tokens: torch.Tensor) -> Iterator[torch.Tensor]:
        """
        Yield feature chunks from a full token sequence.
        tokens : (T, num_codebooks)
        yields : (2*chunk_size, output_dim) tensors
        """
        self.reset()
        T = tokens.shape[0]
        for start in range(0, T, self.chunk_size):
            chunk = tokens[start : start + self.chunk_size]
            yield self.step(chunk)


# ──────────────────────────────────────────────────────────────────────────────
# Latency Benchmark Utility
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_streaming(
    checkpoint: str,
    config: str,
    num_chunks: int = 100,
    chunk_size: int = 50,
    warmup: int = 5,
):
    """Quick per-chunk latency benchmark for streaming mode."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    num_codebooks = cfg["model"]["num_codebooks"]   # read from config, not hardcoded
    output_dim    = cfg["model"]["output_dim"]

    stream = StreamingBridgeInference(checkpoint, config, chunk_size=chunk_size)
    # Build dummy tokens using the correct number of codebooks from config
    dummy_tokens = torch.randint(0, cfg["model"]["vocab_size"], (chunk_size, num_codebooks))

    # Warmup
    stream.reset()
    for _ in range(warmup):
        stream.step(dummy_tokens)

    # Benchmark
    stream.reset()
    times = []
    for _ in range(num_chunks):
        t0 = time.perf_counter()
        stream.step(dummy_tokens)
        if stream.device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    import statistics
    audio_secs_per_chunk = chunk_size / cfg["data"]["mimi_rate"]
    print(f"\n=== Streaming Latency Benchmark ===")
    print(f"Chunk size  : {chunk_size} Mimi frames → {chunk_size * 2} feature frames")
    print(f"Audio / chunk: {audio_secs_per_chunk:.2f}s  |  output_dim={output_dim}")
    print(f"Median : {statistics.median(times):.2f} ms")
    print(f"P95    : {sorted(times)[int(0.95 * len(times))]:.2f} ms")
    print(f"Max    : {max(times):.2f} ms")
    print(f"Throughput: {1000 / statistics.median(times):.1f} chunks/s  "
          f"({1000 * audio_secs_per_chunk / statistics.median(times):.1f}× realtime)")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Mimi-to-HuBERT Bridge Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--config",     required=True,
                        help="Path to config.yaml")
    parser.add_argument("--audio",      default=None,
                        help="Input audio file (.wav / .flac)")
    parser.add_argument("--tokens",     default=None,
                        help="Pre-extracted .pt token file (T, num_codebooks)")
    parser.add_argument("--output",     default="features.pt",
                        help="Output .pt file path (non-compare mode)")
    parser.add_argument("--streaming",  action="store_true",
                        help="Use causal streaming mode (KV-cache)")
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="Chunk size in Mimi frames (streaming only)")
    parser.add_argument("--benchmark",  action="store_true",
                        help="Run latency benchmark instead of inference")
    parser.add_argument("--device",     default=None,
                        help="Force device (cuda / cpu)")

    # ── Compare mode ──────────────────────────────────────────────────────────
    parser.add_argument("--compare",    action="store_true",
                        help=(
                            "Compare real HuBERT output (ONNX) vs bridge model prediction "
                            "for the given --audio file. Prints MSE/MAE/RMSE/cosine/SNR. "
                            "Automatically saves bridge_pred_features.npy and "
                            "hubert_gt_features.npy. Requires --audio."
                        ))
    parser.add_argument("--hubert-model", default=None,
                        help="(compare mode) Override path to HuBERT ONNX file")
    parser.add_argument("--mimi-model",   default=None,
                        help="(compare mode) Override Mimi HF repo or local path")
    parser.add_argument("--save-gt",      default=None,
                        help="(compare mode) Also save ground-truth features as .pt")
    parser.add_argument("--save-pred",    default=None,
                        help="(compare mode) Also save bridge prediction features as .pt")
    # ── .npy output paths (compare mode) ─────────────────────────────────────
    parser.add_argument("--save-gt-npy",   default="hubert_gt_features.npy",
                        help="(compare mode) Path for HuBERT GT .npy output "
                             "[default: hubert_gt_features.npy]")
    parser.add_argument("--save-pred-npy", default="bridge_pred_features.npy",
                        help="(compare mode) Path for Bridge pred .npy output "
                             "[default: bridge_pred_features.npy]")
    parser.add_argument("--no-auto-save-npy", action="store_true",
                        help="(compare mode) Disable automatic .npy saving")
    parser.add_argument("--plot",         action="store_true",
                        help="(compare mode) Show matplotlib heatmap plots")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.benchmark:
        benchmark_streaming(args.checkpoint, args.config, chunk_size=args.chunk_size)
        return

    # ── Compare mode: delegate to compare_inference.compare() ─────────────────
    if args.compare:
        if args.audio is None:
            parser.error("--compare requires --audio")
        from compare_inference import compare as run_compare
        run_compare(
            audio_path            = args.audio,
            checkpoint_path       = args.checkpoint,
            config_path           = args.config,
            device                = args.device,
            hubert_model_override = args.hubert_model,
            mimi_model_override   = args.mimi_model,
            save_gt               = args.save_gt,
            save_pred             = args.save_pred,
            save_gt_npy           = args.save_gt_npy,
            save_pred_npy         = args.save_pred_npy,
            auto_save_npy         = not args.no_auto_save_npy,
            plot                  = args.plot,
        )
        return

    if args.audio is None and args.tokens is None:
        parser.error("Provide at least one of --audio or --tokens")

    # ── Load tokens (from file or by running Mimi extractor) ─────────────────
    def get_tokens(cfg: dict) -> torch.Tensor:
        if args.tokens:
            try:
                return torch.load(args.tokens, weights_only=True)
            except TypeError:
                return torch.load(args.tokens)
        # Extract on-the-fly from audio at native sample rate.
        # MimiExtractor.extract() will resample to 24,000 Hz internally.
        from dataset import MimiExtractor
        import torchaudio
        wav, native_sr = torchaudio.load(args.audio)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        # Do NOT pre-resample to 16 kHz — Mimi requires 24 kHz (handled internally).
        return MimiExtractor(cfg["paths"]["mimi_model"]).extract(wav, native_sr)

    if args.streaming:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        tokens = get_tokens(cfg)

        stream = StreamingBridgeInference(
            args.checkpoint, args.config,
            chunk_size=args.chunk_size, device=args.device,
        )
        chunks   = list(stream.stream_tokens(tokens))
        features = torch.cat(chunks, dim=0)   # (T_h, output_dim)
        logger.info(f"Streaming output: {features.shape}")

    else:
        infer = BridgeInference(args.checkpoint, args.config, device=args.device)
        if args.audio:
            features = infer.from_audio(args.audio)
        else:
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            tokens   = get_tokens(cfg)
            features = infer(tokens).squeeze(0)
        logger.info(f"Batch output: {features.shape}")

    torch.save(features, args.output)
    logger.info(f"Saved features → {args.output}")


if __name__ == "__main__":
    main()
