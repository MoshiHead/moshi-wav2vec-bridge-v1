"""
preprocess.py — Data Preparation Scripts  (Multi-GPU Edition)
=============================================================
Prepares paired (Mimi tokens, Wav2Vec2 features, prosody) from raw audio datasets.

Teacher model: facebook/wav2vec2-base-960h (768-dim, ~50 Hz native → 25 Hz target).

Supports:
  - LibriSpeech  (train-clean-100, train-clean-360, train-other-500, test-clean)
  - VoxCeleb1/2  (flat directory structure)
  - Generic      (any directory of .wav / .flac files)

Outputs:
  - data/train.jsonl
  - data/val.jsonl
  - (optionally) pre-cached feature tensors in data/cache/

Multi-GPU Usage (4× RTX 4090, RunPod):
  # Recommended — torchrun launches one process per GPU automatically
  torchrun --nproc_per_node=4 preprocess.py \
      --dataset librispeech \
      --root /data/LibriSpeech/train-clean-100 \
      --out_dir data \
      --preextract \
      --device cuda \
      --num_workers 4

  # Single-GPU / CPU (unchanged behaviour)
  python preprocess.py \
      --dataset generic \
      --root /data/my_audio \
      --out_dir data
"""

import argparse
import json
import logging
import math
import os
import queue
import random
import threading
from pathlib import Path
from typing import List, Tuple

import yaml

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ──────────────────────────────────────────────────────────────────────────────

def _dist_info() -> Tuple[int, int]:
    """Return (local_rank, world_size).  Works with torchrun and plain python."""
    rank  = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world


def _is_main() -> bool:
    return _dist_info()[0] == 0


def _init_dist():
    """Initialise process group when launched with torchrun (WORLD_SIZE > 1)."""
    rank, world = _dist_info()
    if world > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        return True
    return False


def _barrier():
    """Synchronise all ranks (no-op in single-process mode)."""
    _, world = _dist_info()
    if world > 1:
        import torch.distributed as dist
        dist.barrier()


def _shard_list(items: list, rank: int, world: int) -> list:
    """Evenly divide `items` across `world` ranks; rank gets its slice."""
    return items[rank::world]


# ──────────────────────────────────────────────────────────────────────────────
# Audio file discovery
# ──────────────────────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus"}


def discover_audio(root: str) -> List[Path]:
    """Recursively find all audio files under root."""
    found = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in AUDIO_EXTENSIONS:
            found.append(p)
    found.sort()
    return found


def discover_librispeech(root: str) -> List[Tuple[Path, str]]:
    """
    Yields (audio_path, transcript) tuples.
    LibriSpeech structure: SPEAKER/CHAPTER/SPEAKER-CHAPTER-UUUU.flac + .trans.txt
    """
    pairs = []
    root_p = Path(root)
    for trans_file in root_p.rglob("*.trans.txt"):
        chapter_dir = trans_file.parent
        with open(trans_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts  = line.split(" ", 1)
                utt_id = parts[0]
                text   = parts[1] if len(parts) > 1 else ""
                audio  = chapter_dir / f"{utt_id}.flac"
                if audio.exists():
                    pairs.append((audio, text))
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Feature pre-extraction — multi-GPU sharded
# ──────────────────────────────────────────────────────────────────────────────

def preextract_features(
    audio_paths: List[Path],
    cfg: dict,
    cache_dir: Path,
    device_str: str = "cuda",
    batch_size: int = 1,
    num_workers: int = 4,
):
    """
    Pre-extract and cache Mimi tokens + HuBERT features.

    Multi-GPU strategy
    ──────────────────
    When launched with ``torchrun --nproc_per_node=N``:
      - Each rank handles a disjoint shard of the file list  (rank-i processes
        files[i::N]).  No GPU-to-GPU communication is required; this is purely
        embarrassingly-parallel.
      - Each rank binds to its own CUDA device  (cuda:0 … cuda:3).
      - Rank 0 prints progress and summary; other ranks log to per-rank files.
      - A barrier at the end ensures all ranks finish before the main process
        continues.

    4× RTX 4090 specifics
    ─────────────────────
    - Each 4090 has 24 GB VRAM — models + audio fit easily.
    - ``num_workers`` I/O threads per rank keep each GPU saturated.
    - ``pin_memory=True`` + ``non_blocking=True`` GPU transfers maximise PCIe
      throughput (4090 uses PCIe 4.0 x16 on most RunPod nodes).
    """
    import hashlib
    import torch
    import torchaudio
    from dataset import MimiExtractor, Wav2Vec2Extractor  # project-local

    rank, world = _dist_info()

    # ── Per-rank device assignment ────────────────────────────────────────────
    # Support explicit "cuda:N", bare "cuda" (auto-assign by rank), or "cpu".
    if device_str == "cuda" or device_str == "cuda:0":
        if torch.cuda.is_available():
            device = f"cuda:{rank % torch.cuda.device_count()}"
        else:
            device = "cpu"
    else:
        device = device_str   # user supplied explicit "cuda:2", "cpu", etc.

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)

    # NOTE: We no longer pre-resample audio here. Each extractor handles its own
    # required sample rate internally:
    #   - HuBERTExtractor.extract() resamples to 16,000 Hz (Ditto pipeline).
    #   - MimiExtractor.extract()   resamples to 24,000 Hz (Mimi requirement).
    # This ensures audio quality is preserved for both pipelines.

    # ── Per-rank logging setup ────────────────────────────────────────────────
    rank_logger = logging.getLogger(f"rank{rank}")
    if not _is_main():
        log_path = cache_dir.parent / "logs" / f"rank{rank}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        rank_logger.addHandler(fh)
        rank_logger.setLevel(logging.INFO)
        rank_logger.propagate = False

    rank_logger.info(
        f"[rank {rank}/{world}] device={device}, "
        f"total_files={len(audio_paths)}, io_workers={num_workers}"
    )

    # ── Shard the work ────────────────────────────────────────────────────────
    my_files = _shard_list(audio_paths, rank, world)
    rank_logger.info(f"[rank {rank}] shard size = {len(my_files)} files")

    mimi_ext     = MimiExtractor(cfg["paths"]["mimi_model"], device)
    wav2vec2_ext = Wav2Vec2Extractor(
        cfg["paths"]["wav2vec2_model"], device
    )

    cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(p, suffix):
        h = hashlib.md5(str(p).encode()).hexdigest()
        return cache_dir / f"{h}_{suffix}.pt"

    # ── Filter cached files ────────────────────────────────────────────────────
    todo: List[Path] = []
    n_skipped = 0
    for ap in my_files:
        cp_m  = cache_path(ap, "mimi")
        cp_w2 = cache_path(ap, "wav2vec2")
        fully_cached = (
            cp_m.exists() and cp_w2.exists()
            and cp_m.stat().st_size > 0
            and cp_w2.stat().st_size > 0
        )
        if fully_cached:
            n_skipped += 1
        else:
            if cp_m.exists() or cp_w2.exists():
                rank_logger.warning(
                    f"Partial cache for {ap.name}; re-extracting."
                )
            cp_m.unlink(missing_ok=True)
            cp_w2.unlink(missing_ok=True)
            todo.append(ap)

    rank_logger.info(
        f"[rank {rank}] {n_skipped}/{len(my_files)} already cached — "
        f"extracting {len(todo)} files"
    )

    if not todo:
        rank_logger.info(f"[rank {rank}] Nothing to extract.")
        _barrier()
        return

    # ── Parallel I/O queue ────────────────────────────────────────────────────
    _SENTINEL = None
    load_q: "queue.Queue" = queue.Queue(maxsize=num_workers * 2)

    def _load_worker(paths: List[Path]):
        for ap in paths:
            try:
                wav, file_sr = torchaudio.load(str(ap))
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                # Keep audio at native sample rate — each extractor resamples
                # internally to its required rate (HuBERT→16kHz, Mimi→24kHz).
                if use_cuda:
                    wav = wav.pin_memory()
                load_q.put((ap, wav, file_sr))
            except Exception as e:
                rank_logger.warning(f"Load failed {ap}: {e}")
                load_q.put((ap, None, None))
        load_q.put(_SENTINEL)

    chunk_size    = max(1, math.ceil(len(todo) / num_workers))
    worker_chunks = [todo[i : i + chunk_size] for i in range(0, len(todo), chunk_size)]

    threads = []
    for chunk in worker_chunks:
        t = threading.Thread(target=_load_worker, args=(chunk,), daemon=True)
        t.start()
        threads.append(t)

    # ── GPU extraction loop ───────────────────────────────────────────────────
    import torch

    n_done = n_failed = sentinels = 0
    n_workers = len(threads)

    # tqdm only on rank-0 to avoid interleaved output
    pbar = None
    if _is_main():
        try:
            from tqdm import tqdm
            pbar = tqdm(
                total=len(todo) * world,   # approximate global total
                desc=f"Extracting [{world} GPU(s)]",
                unit="file",
                dynamic_ncols=True,
            )
        except ImportError:
            pass

    while sentinels < n_workers:
        item = load_q.get()
        if item is _SENTINEL:
            sentinels += 1
            continue

        ap, wav, file_sr = item
        cp_m  = cache_path(ap, "mimi")
        cp_w2 = cache_path(ap, "wav2vec2")

        if wav is None:
            n_failed += 1
        else:
            try:
                if use_cuda:
                    wav = wav.to(device, non_blocking=True)

                with torch.no_grad():
                    # MimiExtractor  → resamples to 24,000 Hz internally
                    # Wav2Vec2Extractor → resamples to 16,000 Hz internally
                    tokens   = mimi_ext.extract(wav, file_sr)      # (T_m, 8) at 12.5 Hz
                    w2_feats = wav2vec2_ext.extract(wav, file_sr)  # (T_h, 768) at 25 Hz

                torch.save(tokens,   cp_m)
                torch.save(w2_feats, cp_w2)
                n_done += 1

            except Exception as e:
                rank_logger.warning(f"Extraction failed {ap}: {e}")
                n_failed += 1
                cp_m.unlink(missing_ok=True)
                cp_w2.unlink(missing_ok=True)

        if pbar:
            pbar.update(world)  # each GPU processed one file simultaneously
            pbar.set_postfix(
                rank0_done=n_done,
                rank0_fail=n_failed,
                skipped=n_skipped,
            )
        elif (n_done + n_failed) % 100 == 0:
            rank_logger.info(
                f"[rank {rank}] {n_done+n_failed}/{len(todo)} — "
                f"done={n_done} skip={n_skipped} fail={n_failed}"
            )

    if pbar:
        pbar.close()

    for t in threads:
        t.join()

    rank_logger.info(
        f"[rank {rank}] Done — new={n_done}, cached={n_skipped}, "
        f"failed={n_failed}"
    )

    # ── Wait for all ranks before returning ───────────────────────────────────
    _barrier()

    if _is_main():
        logger.info(
            f"Pre-extraction complete across {world} GPU(s). "
            f"Rank-0: new={n_done}, skipped={n_skipped}, failed={n_failed}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Manifest writers
# ──────────────────────────────────────────────────────────────────────────────

def write_manifest(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info(f"Wrote {len(records)} records → {path}")


def build_manifests(
    audio_pairs: List[Tuple[Path, str]],
    out_dir: Path,
    val_frac: float = 0.01,
    seed: int = 42,
):
    """Split into train/val manifests.  Only rank-0 writes files."""
    if not _is_main():
        _barrier()          # non-main ranks wait for rank-0 to finish writing
        return

    rng   = random.Random(seed)
    pairs = list(audio_pairs)
    rng.shuffle(pairs)

    n_val = max(1, int(len(pairs) * val_frac))
    val   = pairs[:n_val]
    train = pairs[n_val:]

    def to_record(audio_path, text):
        return {"audio_path": str(audio_path), "text": text}

    write_manifest([to_record(p, t) for p, t in train], out_dir / "train.jsonl")
    write_manifest([to_record(p, t) for p, t in val],   out_dir / "val.jsonl")
    logger.info(f"Train: {len(train)} | Val: {len(val)}")

    _barrier()              # signal non-main ranks they can proceed


# ──────────────────────────────────────────────────────────────────────────────
# Main CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess audio data for bridge training (multi-GPU)"
    )
    parser.add_argument(
        "--dataset", choices=["librispeech", "voxceleb", "generic"],
        default="generic",
    )
    parser.add_argument("--root",     required=True, help="Root directory of audio")
    parser.add_argument("--out_dir",  default="data", help="Output dir for manifests")
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--val_frac", type=float, default=0.01)
    parser.add_argument(
        "--preextract", action="store_true",
        help="Pre-extract and cache Mimi + Wav2Vec2 features",
    )
    parser.add_argument(
        "--device", default="cuda",
        help=(
            '"cuda"  → each rank auto-assigns cuda:rank  (recommended for RunPod)\n'
            '"cuda:N" → pin all ranks to one GPU\n'
            '"cpu"   → CPU-only mode'
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="I/O threads per GPU rank for parallel audio loading",
    )
    args = parser.parse_args()

    # ── Distributed init ──────────────────────────────────────────────────────
    _init_dist()
    rank, world = _dist_info()

    log_level = logging.INFO if _is_main() else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format=f"[rank {rank}] %(asctime)s %(levelname)s %(message)s",
    )

    if _is_main():
        logger.info(f"Running with {world} process(es) / GPU(s)")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir)

    # ── Discover files (all ranks do discovery; it's read-only + fast) ────────
    if _is_main():
        logger.info(f"Discovering {args.dataset} audio under {args.root}")

    if args.dataset == "librispeech":
        pairs = discover_librispeech(args.root)
        if _is_main():
            logger.info(f"Found {len(pairs)} LibriSpeech utterances")
    else:
        audio_files = discover_audio(args.root)
        pairs = [(p, "") for p in audio_files]
        if _is_main():
            logger.info(f"Found {len(pairs)} audio files")

    if not pairs:
        if _is_main():
            logger.error("No audio files found. Check --root path.")
        return

    # ── Build manifests (rank-0 only, others wait at barrier) ─────────────────
    build_manifests(pairs, out_dir, val_frac=args.val_frac, seed=args.seed)

    # ── Optional pre-extraction (all ranks participate) ───────────────────────
    if args.preextract:
        if _is_main():
            logger.info(
                f"Pre-extracting features across {world} GPU(s). "
                f"Each GPU handles ~{len(pairs)//world} files."
            )
        cache_dir  = Path(cfg["data"]["cache_dir"])
        audio_only = [p for p, _ in pairs]
        preextract_features(
            audio_only, cfg, cache_dir,
            device_str=args.device,
            num_workers=args.num_workers,
        )

    if _is_main():
        logger.info("Preprocessing done.")


if __name__ == "__main__":
    main()