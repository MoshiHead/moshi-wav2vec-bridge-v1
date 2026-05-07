# Mimi-to-HuBERT Bridge Module

A complete training and inference system that converts **Mimi discrete token streams** (12.5 Hz, 8-codebook) into **HuBERT-compatible continuous features** (50 Hz, 768-dim) for use in downstream diffusion-based talking-head models such as Ditto.

---

## System Overview

```
Raw Audio
    │
    ▼
[Mimi Encoder]  ──→  tokens (B, T, 8)  at 12.5 Hz
  HuggingFace MimiModel
  kyutai/moshiko-pytorch-bf16
  (no moshi git install required)
    │
    ▼
[Bridge Module]
    ├─ Multi-codebook Embeddings  (B, T, 256)
    ├─ CausalUpsample ×4          (B, 4T, 512)
    ├─ Causal Transformer          (B, 4T, 512)
    └─ Output Projection           (B, 4T, 1024)
    │
    ▼
Predicted Features  (B, 4T, 1024)  at 50 Hz
    │
    ▼ (during training only)
Multi-Loss vs HuBERT-large targets  (facebook/hubert-large-ls960-ft, 1024-dim)
```

### Multi-GPU training

The trainer auto-detects available GPUs:

```bash
# Single GPU (default)
python train.py --config config.yaml

# DataParallel (multiple GPUs, no launcher needed)
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config config.yaml

# DistributedDataParallel (fastest, requires torchrun)
torchrun --nproc_per_node=4 train.py --config config.yaml
```

---

## Repository Structure

```
mimi_hubert_bridge/
├── config.yaml       # All hyperparameters and paths
├── model.py          # Bridge architecture + Discriminator
├── losses.py         # All 7 loss functions + BridgeLoss manager
├── dataset.py        # Data loading, Mimi/HuBERT extraction, prosody
├── trainer.py        # Full training loop with AMP, logging, checkpointing
├── inference.py      # Batch and streaming inference + CLI
├── preprocess.py     # Data preparation scripts
├── train.py          # Main training entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# transformers>=4.40.0 is required for MimiModel (HuggingFace Mimi encoder).
# The moshi git repository is no longer needed.
```

### 2. Prepare data

```bash
#multiple_GPU_use
!torchrun --nproc_per_node=4 preprocess.py \
    --dataset librispeech \
    --root ./data/LibriSpeech/train-clean-100 \
    --out_dir data \
    --val_frac 0.1 \
    --preextract \
    --device cuda \
    --num_workers 16

# From LibriSpeech
python preprocess.py \
    --dataset librispeech \
    --root /data/LibriSpeech/train-clean-100 \
    --out_dir data \
    --val_frac 0.01 \
    --preextract \
    --device cuda

# From any directory of audio files
python preprocess.py \
    --dataset generic \
    --root /data/my_audio \
    --out_dir data
```

Outputs `data/train.jsonl` and `data/val.jsonl`. Each line:
```json
{"audio_path": "/path/to/file.flac", "text": "optional transcript"}
```

### 3. Configure

Edit `config.yaml`. Key settings:

```yaml
paths:
  hubert_model: "facebook/hubert-base-ls960"
  mimi_model:   "kyutai/mimi"

training:
  batch_size: 16
  num_epochs: 50
  loss_weights:
    recon:     1.0
    ctc:       0.3
    prosody:   0.2
    adv:       0.1
    stat:      0.1
    smooth:    0.05
    alignment: 0.1
```

### 4. Train

```bash
python train.py --config config.yaml

# Resume from checkpoint
python train.py --config config.yaml --resume checkpoints/bridge_best.pt

# Override hyperparameters on the fly
python train.py --config config.yaml --overrides training.batch_size=8 training.learning_rate=5e-5
```

### 5. Inference

**Batch mode:**
```bash
python inference.py \
    --checkpoint checkpoints/bridge_best.pt \
    --config config.yaml \
    --audio input.wav \
    --output features.pt
```

**Streaming mode** (causal, chunk-by-chunk):
```bash
python inference.py \
    --checkpoint checkpoints/bridge_best.pt \
    --config config.yaml \
    --audio input.wav \
    --output features.pt \
    --streaming \
    --chunk-size 50
```

**Latency benchmark:**
```bash
python inference.py \
    --checkpoint checkpoints/bridge_best.pt \
    --config config.yaml \
    --benchmark
```

**Compare Loss:**
```bash
!python inference.py \
  --checkpoint checkpoints/bridge_best.pt \
  --config config.yaml \
  --audio /content/audio.wav \
  --compare
```

**Python API:**
```python
from inference import BridgeInference, StreamingBridgeInference

# Batch
infer = BridgeInference("checkpoints/bridge_best.pt", "config.yaml")
features = infer.from_audio("speech.wav")   # (T_h, 768)

# Streaming
stream = StreamingBridgeInference("checkpoints/bridge_best.pt", "config.yaml", chunk_size=50)
stream.reset()
for chunk in token_stream:   # each chunk: (50, 8)
    feat = stream.step(chunk) # (200, 768) — no future frames used
```

---

## Architecture Details

### Token Embedding
- 8 **separate** `nn.Embedding` tables (one per codebook), vocab size 2048
- Per-level learnable scale weights
- Fusion: element-wise **sum** → `(B, T, 256)` (or `concat` + linear)

### Causal Upsampler (×4)
- `ConvTranspose1d(stride=4, kernel=4)` — clean 4× expansion
- Followed by a **left-padded** causal Conv1d (no future frames)
- GroupNorm + GELU activation

### Causal Transformer
- Pre-norm residual architecture
- **T5-style relative position bias** — better than sinusoidal for variable-length speech
- Causal mask enforced in attention (upper triangle = −∞)
- **KV-cache** for O(1) per-step inference in streaming mode

### Output Projection
```
Linear(512→512) → GELU → Linear(512→768)
```

---

## Loss Functions

| # | Name | Component | Default Weight |
|---|------|-----------|----------------|
| 1 | **Reconstruction** | MSE + cosine distance vs HuBERT | 1.0 |
| 2 | **CTC** | Frozen ASR head phonetic intelligibility | 0.3 |
| 3 | **Prosody** | Predicted F0 + energy MSE | 0.2 |
| 4 | **Adversarial** | Hinge GAN (delayed start at step 5000) | 0.1 |
| 5 | **Statistics** | Mean + variance distribution matching | 0.1 |
| 6 | **Smoothness** | First-order temporal difference penalty | 0.05 |
| 7 | **Alignment** | Frame-level phoneme cross-entropy | 0.1 |

---

## Training Notes

- Adversarial training starts at `disc_start_step` (default 5000) to stabilise early training
- Mixed precision (BF16/FP16) enabled by default on CUDA
- Cosine LR schedule with linear warmup
- Separate AdamW optimisers for generator and discriminator
- Checkpoints saved every epoch + best-val separately

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `recon_mse` | L2 distance in HuBERT feature space |
| `pitch_corr` | Pearson correlation of predicted vs GT F0 |
| `ctc` | CTC loss (proxy for phonetic intelligibility) |
| `stat_mean/std` | Distribution mismatch |
| `smooth` | Temporal discontinuity |

---

## Drop-in Replacement for HuBERT

The bridge output `(B, T, 768)` at 50 Hz is directly compatible with any model trained on HuBERT features (e.g. Ditto, SadTalker, DiffTalk). Simply replace the HuBERT feature extractor call:

```python
# Before
hubert_features = hubert_model(audio)   # slow, requires audio

# After
hubert_features = bridge(mimi_tokens)   # fast, causal, streamable
```

---

## Hardware Requirements

| Mode | VRAM | Notes |
|------|------|-------|
| Training (batch=16) | ~12 GB | Single A100/RTX 3090 |
| Training (batch=4)  | ~4 GB  | Consumer GPU |
| Inference (streaming) | <1 GB | CPU viable |

---

## Citing

If you use this bridge module in your work, please cite the underlying models:
- Mimi: Défossez et al., "Moshi: a speech-text foundation model for real-time dialogue" (2024)
- HuBERT: Hsu et al., "HuBERT: Self-Supervised Speech Representation Learning" (2021)
