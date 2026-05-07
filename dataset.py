"""
dataset.py — Data Loading & Feature Extraction
===============================================
Handles:
  - Audio loading and resampling
  - Mimi tokenization via HuggingFace (kyutai/moshiko-pytorch-bf16) at 12.5 Hz
    ► Mimi ALWAYS receives 24,000 Hz audio (resampled internally if needed).
  - Ditto-compatible HuBERT feature extraction (ONNX streaming, 25 Hz).
    The extraction pipeline exactly mirrors Ditto's Wav2FeatHubert.wav2feat():
      • audio_16k = resample(audio, orig_sr, 16000)  [via torchaudio / librosa]
      • Sliding window chunks with left/right zero-padding
      • Batched ONNX inference   → (N_chunks, frames, 1024)
      • Slice valid frames + mean-pool pairs of 20 ms → (T, 1024) at 25 Hz
    Output is 25 Hz — matching Ditto exactly. NO upsampling is applied.
    Bridge model output rate: Mimi 12.5 Hz × upsample_factor 2 = 25 Hz. ✓
  - Pitch (F0) and energy extraction via pyworld / librosa
  - Optional forced-alignment labels
  - Caching of pre-extracted features
  - Collation with padding masks
"""

import os
import json
import math
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Optional heavy imports (guarded so tests can import dataset.py headlessly)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import torchaudio
    TORCHAUDIO_OK = True
except ImportError:
    TORCHAUDIO_OK = False
    logger.warning("torchaudio not found – audio loading will fail at runtime.")

try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False
    logger.warning("librosa not found – pitch extraction may fail.")

try:
    import pyworld
    PYWORLD_OK = True
except ImportError:
    PYWORLD_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# Feature Extractors (wrappers around heavy models)
# ──────────────────────────────────────────────────────────────────────────────

class MimiExtractor:
    """
    Loads the Mimi audio tokenizer from kyutai/moshiko-pytorch-bf16 on HuggingFace.

    The kyutai/moshiko-pytorch-bf16 repo does NOT contain a preprocessor_config.json
    or processor_config.json, so AutoFeatureExtractor / MimiModel.from_pretrained()
    will always fail for it.  Instead we download the dedicated Mimi weights file
    (tokenizer-e351c8d8-checkpoint125.safetensors) via hf_hub_download and build
    the model with the moshi library, exactly as the reference loader.py does.

    Two back-ends are attempted in order:
      1. moshi  — the official Kyutai Python package (pip install moshi).
         Uses hf_hub_download + safetensors, no preprocessor_config required.
      2. transformers MimiModel  — works when model_name points to a repo that
         *does* have preprocessor_config.json (e.g. kyutai/mimi).
      3. Dummy  — random tokens so the rest of the pipeline stays runnable.

    IMPORTANT — Sample Rate:
        Mimi ALWAYS operates at 24,000 Hz.  The extract() method accepts audio
        at any sample rate and resamples to 24 kHz internally before encoding.
        Callers do NOT need to pre-resample to 24 kHz.

    Install (recommended):
        pip install moshi safetensors huggingface_hub
    Alternative (only for repos with HF processor configs):
        pip install transformers>=4.40.0
    """

    # File name of the Mimi weights inside the kyutai/moshiko-pytorch-bf16 repo
    _MIMI_SAFETENSORS = "tokenizer-e351c8d8-checkpoint125.safetensors"
    # Number of codebooks we ask Mimi to use (matches config.yaml num_codebooks)
    NUM_CODEBOOKS = 8
    # Mimi ALWAYS operates at 24 kHz; hop = 1920 samples → 12.5 Hz frame rate
    _MIMI_SR = 24000
    _HOP = 1920   # 24000 / 12.5

    def __init__(self, model_name: str = "kyutai/moshiko-pytorch-bf16", device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        self._ok = False
        self._backend = None   # "moshi" | "transformers"

        # ── Backend 1: moshi library (handles raw safetensors repos) ──────────
        if not self._ok:
            self._try_load_moshi(model_name, device)

        # ── Backend 2: transformers MimiModel (needs preprocessor_config) ────
        if not self._ok:
            self._try_load_transformers(model_name, device)

        if not self._ok:
            logger.warning(
                f"[MimiExtractor] All loading strategies failed for '{model_name}'. "
                "Running with a DUMMY extractor (random tokens). "
                "To fix: pip install moshi safetensors huggingface_hub"
            )

    # ── Private loader helpers ────────────────────────────────────────────────

    def _try_load_moshi(self, model_name: str, device: str):
        """
        Download tokenizer-e351c8d8-checkpoint125.safetensors from HF and build
        a MimiModel using the moshi library's get_mimi() helper and the
        hard-coded architecture config (_mimi_config) from loader.py.
        """
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file as sf_load
            import torch as _torch

            logger.info(
                f"[MimiExtractor] Downloading Mimi weights from {model_name} "
                f"({self._MIMI_SAFETENSORS}) via hf_hub_download …"
            )
            weights_path = hf_hub_download(
                repo_id=model_name,
                filename=self._MIMI_SAFETENSORS,
            )
            logger.info(f"[MimiExtractor] Weights cached at {weights_path}")

            # Build the Mimi model using the moshi library
            try:
                from moshi.models.loaders import get_mimi
                self.model = get_mimi(weights_path, device=device)
                self.model.set_num_codebooks(self.NUM_CODEBOOKS)
                self.model.eval()
                self._backend = "moshi"
                self._ok = True
                logger.info(
                    f"[MimiExtractor] Loaded via moshi library "
                    f"(num_codebooks={self.NUM_CODEBOOKS})."
                )
                return
            except ImportError:
                logger.info(
                    "[MimiExtractor] moshi package not installed; "
                    "attempting manual safetensors load with built-in architecture …"
                )

            # ── Fallback: build architecture manually without the moshi package ──
            # Architecture constants copied directly from loader.py (_mimi_config).
            self._weights_path = weights_path
            self._ok = True
            self._backend = "safetensors_raw"
            self._sf_path = weights_path
            self._build_moshi_manual(weights_path, device)

        except Exception as e:
            logger.info(f"[MimiExtractor] moshi/safetensors strategy failed: {e}")

    def _build_moshi_manual(self, weights_path: str, device: str):
        """
        Instantiate Mimi manually using only torch + safetensors, without the
        moshi package. Uses the exact architecture kwargs from loader.py.
        Falls back to transformers if this also fails.
        """
        try:
            from safetensors.torch import load_file as sf_load

            # These are the exact kwargs from loader.py
            seanet_kwargs = {
                "channels": 1, "dimension": 512, "causal": True,
                "n_filters": 64, "n_residual_layers": 1, "activation": "ELU",
                "compress": 2, "dilation_base": 2,
                "disable_norm_outer_blocks": 0, "kernel_size": 7,
                "residual_kernel_size": 3, "last_kernel_size": 3,
                "norm": "none", "pad_mode": "constant",
                "ratios": [8, 6, 5, 4], "true_skip": True,
            }
            quantizer_kwargs = {
                "dimension": 256, "n_q": 32, "bins": 2048,
                "input_dimension": seanet_kwargs["dimension"],
                "output_dimension": seanet_kwargs["dimension"],
            }
            transformer_kwargs = {
                "d_model": seanet_kwargs["dimension"], "num_heads": 8,
                "num_layers": 8, "causal": True, "layer_scale": 0.01,
                "context": 250, "conv_layout": True, "max_period": 10000,
                "gating": "none", "norm": "layer_norm",
                "positional_embedding": "rope", "dim_feedforward": 2048,
                "input_dimension": seanet_kwargs["dimension"],
                "output_dimensions": [seanet_kwargs["dimension"]],
            }

            from moshi.modules import SEANetEncoder, SEANetDecoder
            from moshi.modules import transformer as moshi_transformer
            from moshi.quantization import SplitResidualVectorQuantizer
            from moshi.models.compression import MimiModel

            enc = SEANetEncoder(**seanet_kwargs)
            dec = SEANetDecoder(**seanet_kwargs)
            enc_tr = moshi_transformer.ProjectedTransformer(device=device, **transformer_kwargs)
            dec_tr = moshi_transformer.ProjectedTransformer(device=device, **transformer_kwargs)
            quant = SplitResidualVectorQuantizer(**quantizer_kwargs)

            model = MimiModel(
                enc, dec, quant,
                channels=1, sample_rate=24000, frame_rate=12.5,
                encoder_frame_rate=24000 / enc.hop_length,
                causal=True, resample_method="conv",
                encoder_transformer=enc_tr, decoder_transformer=dec_tr,
            ).to(device=device)
            model.eval()

            state = sf_load(weights_path, device=str(device))
            model.load_state_dict(state)
            model.set_num_codebooks(self.NUM_CODEBOOKS)

            self.model = model
            self._backend = "moshi_manual"
            logger.info("[MimiExtractor] Manual moshi architecture load succeeded.")

        except Exception as e:
            logger.warning(
                f"[MimiExtractor] Manual architecture build failed ({e}); "
                "will try transformers backend next."
            )
            self._ok = False

    def _try_load_transformers(self, model_name: str, device: str):
        """
        Fall back to transformers MimiModel + AutoFeatureExtractor.
        Works only for repos that ship preprocessor_config.json (e.g. kyutai/mimi).
        """
        try:
            from transformers import MimiModel, AutoFeatureExtractor
            logger.info(
                f"[MimiExtractor] Trying transformers AutoFeatureExtractor for '{model_name}' …"
            )
            self.processor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = MimiModel.from_pretrained(model_name)
            self.model.eval().to(device)
            self._backend = "transformers"
            self._ok = True
            logger.info("[MimiExtractor] Loaded via transformers MimiModel.")
        except Exception as e:
            logger.info(f"[MimiExtractor] transformers strategy failed: {e}")

    # ── Public interface ──────────────────────────────────────────────────────

    @torch.no_grad()
    def extract(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Encode a waveform into Mimi discrete tokens.

        IMPORTANT: Mimi always requires 24,000 Hz mono audio.  This method
        resamples `wav` from `sr` → 24 kHz automatically, so callers may pass
        audio at any sample rate (e.g. 16 kHz from the HuBERT pipeline).

        wav : (1, samples)  float32  — any supported sample rate
        sr  : sample rate of `wav`

        Returns: (T, NUM_CODEBOOKS) int64 tensor  at 12.5 Hz
        """
        if not self._ok:
            T = max(1, wav.shape[-1] // self._HOP)
            return torch.randint(0, 2048, (T, self.NUM_CODEBOOKS))

        # ── Always resample to 24 kHz for Mimi (regardless of backend) ───────
        # This is the canonical requirement: Mimi expects 24 kHz audio.
        if sr != self._MIMI_SR:
            if TORCHAUDIO_OK:
                wav = torchaudio.functional.resample(wav, sr, self._MIMI_SR)
            else:
                # numpy linear-interpolation fallback (rarely used)
                ratio = self._MIMI_SR / sr
                new_len = int(wav.shape[-1] * ratio)
                wav_np = wav.squeeze(0).cpu().numpy()
                indices = np.linspace(0, len(wav_np) - 1, new_len)
                wav_np = np.interp(indices, np.arange(len(wav_np)), wav_np).astype(np.float32)
                wav = torch.from_numpy(wav_np).unsqueeze(0)
            logger.debug(
                f"[MimiExtractor] Resampled {sr} Hz → {self._MIMI_SR} Hz "
                f"for Mimi encoding."
            )

        # ── moshi / moshi_manual / safetensors_raw backends ──────────────────
        if self._backend in ("moshi", "moshi_manual", "safetensors_raw"):
            # moshi encode() expects (B, C, T); returns codes (B, num_codebooks, T)
            wav_device = wav.to(self.device)
            if wav_device.dim() == 2:
                wav_device = wav_device.unsqueeze(0)   # (1, 1, T)
            elif wav_device.dim() == 1:
                wav_device = wav_device.unsqueeze(0).unsqueeze(0)

            codes = self.model.encode(wav_device)      # (B, num_codebooks, T)
            codes = codes.squeeze(0)                   # (num_codebooks, T)
            codes = codes.transpose(0, 1)              # (T, num_codebooks)
            return codes.cpu().long()

        # ── transformers backend ──────────────────────────────────────────────
        # wav is already at 24 kHz at this point; processor handles its own SR.
        wav_np = wav.squeeze(0).cpu().numpy()
        inputs = self.processor(
            raw_audio=wav_np, sampling_rate=self._MIMI_SR, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        encoder_out = self.model.encode(**inputs)
        codes = encoder_out.audio_codes          # (B, num_codebooks, T)
        codes = codes.squeeze(0).transpose(0, 1)  # (T, num_codebooks)
        return codes.cpu().long()


class Wav2Vec2Extractor:
    """
    HuggingFace-based Wav2Vec2 feature extractor using facebook/wav2vec2-base-960h.

    Pipeline (mirrors the second repo's Wav2VecModel.forward logic):
      1. Resample audio to 16,000 Hz
      2. Run facebook/wav2vec2-base-960h → last_hidden_state (T_native, 768)
         Native rate ≈ 50 Hz  (stride = 320 samples at 16 kHz)
      3. Down-interpolate to 25 Hz:
            num_f = ceil(len_samples_16k / 16000 * 25)
         Uses F.interpolate (linear, align_corners=True) — same as the second
         repo's linear_interpolation() utility.
      4. Returns (num_f, 768) float32 CPU tensor  at 25 Hz

    Bridge output rate: Mimi 12.5 Hz × upsample_factor 2 = 25 Hz ✓

    Args:
        model_name : HuggingFace model ID or local path
                     (default: "facebook/wav2vec2-base-960h")
        device     : "cuda" | "cuda:N" | "cpu"
    """

    _TARGET_SR = 16000   # wav2vec2 expects 16 kHz
    _FEAT_DIM  = 768     # wav2vec2-base hidden size
    _OUT_HZ    = 25      # target output rate (matches bridge output rate)

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        device: str = "cpu",
    ):
        self.device    = device
        self._ok       = False
        self._feat_dim = self._FEAT_DIM

        try:
            from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

            logger.info(
                f"[Wav2Vec2Extractor] Loading {model_name} from HuggingFace …"
            )
            self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self._model = Wav2Vec2Model.from_pretrained(model_name)
            self._model.eval()
            self._model = self._model.to(device)
            self._ok = True
            logger.info(
                f"[Wav2Vec2Extractor] Loaded {model_name} on {device} "
                f"(feat_dim={self._FEAT_DIM}, target_rate={self._OUT_HZ} Hz)."
            )
        except Exception as e:
            logger.warning(
                f"[Wav2Vec2Extractor] Could not load '{model_name}': {e}. "
                "Using dummy extractor (random features). "
                "To fix: pip install transformers>=4.40.0"
            )

    @torch.no_grad()
    def extract(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Extract Wav2Vec2 features down-interpolated to 25 Hz.

        wav : (1, N) float32 torch.Tensor — any sample rate.
        sr  : sample rate of `wav`.

        Returns: (T, 768) float32 torch.Tensor on CPU  at 25 Hz
                 T = ceil(N_16k / 16000 * 25)
        """
        if not self._ok:
            T = max(1, wav.shape[-1] // 640)   # approx 25 Hz at 16 kHz
            return torch.randn(T, self._feat_dim)

        # ── 1. Resample to 16 kHz ────────────────────────────────────────────
        if sr != self._TARGET_SR:
            if TORCHAUDIO_OK:
                wav = torchaudio.functional.resample(wav, sr, self._TARGET_SR)
            elif LIBROSA_OK:
                wav_np = wav.squeeze().cpu().numpy().astype(np.float32)
                wav_np = librosa.resample(wav_np, orig_sr=sr, target_sr=self._TARGET_SR)
                wav = torch.from_numpy(wav_np).unsqueeze(0)
            else:
                raise RuntimeError(
                    "Neither torchaudio nor librosa available for resampling."
                )

        # ── 2. Compute target frame count at 25 Hz ───────────────────────────
        num_f = math.ceil(wav.shape[-1] / self._TARGET_SR * self._OUT_HZ)
        num_f = max(1, num_f)

        # ── 3. Feature extraction via HuggingFace wav2vec2 ───────────────────
        wav_np = wav.squeeze().cpu().numpy().astype(np.float32)
        inputs = self._feature_extractor(
            wav_np, sampling_rate=self._TARGET_SR, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)  # (1, N)

        outputs = self._model(input_values)
        # last_hidden_state: (1, T_native, 768)  at ~50 Hz
        features = outputs.last_hidden_state  # (1, T_native, 768)

        # ── 4. Down-interpolate from ~50 Hz → 25 Hz ──────────────────────────
        # Uses F.interpolate (linear, align_corners=True) — identical to
        # linear_interpolation() in the second repo's wav2vec.py.
        features = features.transpose(1, 2)          # (1, 768, T_native)
        features = F.interpolate(
            features,
            size=num_f,
            mode="linear",
            align_corners=True,
        )                                             # (1, 768, num_f)
        features = features.transpose(1, 2)          # (1, num_f, 768)
        features = features.squeeze(0)               # (num_f, 768)

        return features.cpu().float()




def extract_f0_energy(
    wav: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160,
    f0_min: float = 50.0,
    f0_max: float = 600.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract normalised log-F0 and log-energy using pyworld or librosa fallback.

    Returns:
        f0:      (T,) normalised in [0, 1], 0 = unvoiced
        energy:  (T,) normalised in [0, 1]
        voiced:  (T,) bool
    """
    if PYWORLD_OK:
        _wav = wav.astype(np.float64)
        f0, t = pyworld.harvest(_wav, sr, f0_floor=f0_min, f0_ceil=f0_max,
                                frame_period=hop_length / sr * 1000)
        voiced = f0 > 0
        f0_log = np.where(voiced, np.log(f0 + 1e-8), 0.0)
    elif LIBROSA_OK:
        f0_arr, voiced_flag, _ = librosa.pyin(
            wav, fmin=f0_min, fmax=f0_max, sr=sr, hop_length=hop_length
        )
        f0_arr = np.nan_to_num(f0_arr, nan=0.0)
        voiced = voiced_flag.astype(bool)
        f0_log = np.where(voiced, np.log(f0_arr + 1e-8), 0.0)
    else:
        T = math.ceil(len(wav) / hop_length)
        return np.zeros(T), np.zeros(T), np.zeros(T, dtype=bool)

    # Energy via RMS
    if LIBROSA_OK:
        rms = librosa.feature.rms(y=wav, hop_length=hop_length, frame_length=hop_length * 4)[0]
        rms = rms[:len(f0_log)]
    else:
        # Manual RMS
        frames = [wav[i:i + hop_length * 4] for i in range(0, len(wav), hop_length)]
        rms = np.array([np.sqrt(np.mean(f**2 + 1e-8)) for f in frames])
        rms = rms[:len(f0_log)]

    energy_log = np.log(rms + 1e-8)

    # Normalise to [0, 1]
    def safe_norm(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-8)

    return safe_norm(f0_log).astype(np.float32), safe_norm(energy_log).astype(np.float32), voiced


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class MimiWav2Vec2Dataset(Dataset):
    """
    Paired dataset that provides:
      - Mimi tokens    (T_m, 8)              at 12.5 Hz
      - Wav2Vec2 GT    (T_h, 768)            at 25 Hz  (T_h = 2 * T_m)
        ► Extracted via facebook/wav2vec2-base-960h, down-interpolated from
          ~50 Hz to 25 Hz using F.interpolate (linear, align_corners=True).
          Formula: num_f = ceil(len_samples_16k / 16000 * 25)
      - Prosody        (T_h,) F0 and energy  at 25 Hz
      - Optional phoneme labels
    """

    def __init__(
        self,
        manifest_path: str,
        cfg: dict,
        split: str = "train",
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.split = split
        # Load at 16 kHz (for Wav2Vec2 pipeline); Mimi resamples internally to 24 kHz
        self.sr = cfg["data"]["sample_rate"]          # 16000
        self.mimi_sr = cfg["data"].get("mimi_sample_rate", 24000)   # 24000
        self.max_len = int(cfg["data"]["max_audio_seconds"] * self.sr)
        self.cache_features = cfg["data"].get("cache_features", True)
        self.cache_dir = Path(cfg["data"].get("cache_dir", "data/cache"))
        self.hop_length = cfg["training"].get("hop_length", 160)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load manifest
        with open(manifest_path) as f:
            self.samples = [json.loads(l) for l in f]

        logger.info(f"Loaded {len(self.samples)} samples ({split})")

        # Lazy-init extractors (heavy; only when needed)
        self._mimi = None
        self._wav2vec2 = None
        self._device = device

    def _get_mimi(self):
        if self._mimi is None:
            self._mimi = MimiExtractor(self.cfg["paths"]["mimi_model"], self._device)
        return self._mimi

    def _get_wav2vec2(self):
        if self._wav2vec2 is None:
            self._wav2vec2 = Wav2Vec2Extractor(
                self.cfg["paths"]["wav2vec2_model"], self._device
            )
        return self._wav2vec2

    def _cache_path(self, audio_path: str, suffix: str) -> Path:
        h = hashlib.md5(audio_path.encode()).hexdigest()
        return self.cache_dir / f"{h}_{suffix}.pt"

    def _load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and return (waveform, native_sr).

        The waveform is converted to mono but kept at the native file sample rate.
        Each extractor is responsible for resampling to its required rate:
          - HuBERTExtractor: resamples to 16,000 Hz internally
          - MimiExtractor  : resamples to 24,000 Hz internally

        Returns:
            waveform : (1, samples)  float32  at native file sample rate
            native_sr: int           original sample rate from file
        """
        waveform, native_sr = torchaudio.load(audio_path)
        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        # Trim to max length (in native SR samples to preserve all audio)
        max_native = int(self.cfg["data"]["max_audio_seconds"] * native_sr)
        if waveform.shape[-1] > max_native:
            waveform = waveform[:, :max_native]
        return waveform, native_sr

    def _get_or_cache(self, audio_path: str, key: str, extractor_fn):
        cp = self._cache_path(audio_path, key)
        if self.cache_features and cp.exists():
            try:
                # weights_only=False is required: prosody cache stores a tuple,
                # which weights_only=True refuses to deserialize. All cache files
                # are produced by this codebase so the risk is acceptable.
                return torch.load(cp, map_location="cpu", weights_only=False)
            except Exception:
                cp.unlink(missing_ok=True)   # corrupted — regenerate below
        result = extractor_fn()
        if self.cache_features:
            torch.save(result, cp)
        return result


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        audio_path = sample["audio_path"]

        # --- Load audio at native sample rate ---
        wav, native_sr = self._load_audio(audio_path)  # (1, N) at native_sr

        # For prosody (pyworld/librosa), we need a numpy array at 16 kHz
        if native_sr != self.sr:
            if TORCHAUDIO_OK:
                wav_16k = torchaudio.functional.resample(wav, native_sr, self.sr)
            else:
                wav_16k = wav
        else:
            wav_16k = wav
        wav_np = wav_16k.squeeze().numpy()

        # --- Mimi tokens ---
        tokens = self._get_or_cache(
            audio_path, "mimi",
            lambda: self._get_mimi().extract(wav, native_sr)
        )  # (T_m, 8) at 12.5 Hz

        # --- Wav2Vec2 GT features at 25 Hz ---
        # Wav2Vec2Extractor.extract():
        #   1. Resamples to 16 kHz internally
        #   2. Runs facebook/wav2vec2-base-960h → last_hidden_state (~50 Hz)
        #   3. Down-interpolates to 25 Hz:  num_f = ceil(N_16k / 16000 * 25)
        # Bridge also outputs at 25 Hz: Mimi 12.5 Hz × upsample_factor 2 = 25 Hz.
        wav2vec2_feats = self._get_or_cache(
            audio_path, "wav2vec2",
            lambda: self._get_wav2vec2().extract(wav, native_sr),
        )  # (T_h, 768) at 25 Hz

        T_m = tokens.shape[0]
        T_h = wav2vec2_feats.shape[0]

        # Enforce 2:1 ratio — bridge output (25 Hz) : Mimi tokens (12.5 Hz).
        # T_m * 2 ≈ T_h (may differ by ±1 due to rounding — trim to the minimum).
        T_min = min(T_m, T_h // 2)
        tokens        = tokens[:T_min]               # (T_m, 8) at 12.5 Hz
        wav2vec2_feats = wav2vec2_feats[:T_min * 2]  # (2*T_m, 768) at 25 Hz

        # --- Prosody (F0 + energy) — CACHED to avoid per-epoch pyworld cost ---
        # pyworld/librosa F0 extraction takes 50-200 ms per file. Without caching
        # this runs on CPU for every sample every epoch, starving the GPUs.
        # We cache (f0, energy, voiced) as a tuple with a T_h-keyed suffix so that
        # if T_h changes (e.g. after rate changes) the cache is regenerated cleanly.
        T_h = T_min * 2

        def _extract_prosody():
            f0_np, energy_np, voiced_np = extract_f0_energy(
                wav_np, self.sr, self.hop_length
            )
            f0_r     = torch.from_numpy(self._resample_array(f0_np,     T_h))
            energy_r = torch.from_numpy(self._resample_array(energy_np, T_h))
            voiced_r = torch.from_numpy(
                self._resample_array(voiced_np.astype(np.float32), T_h) > 0.5
            )
            return (f0_r, energy_r, voiced_r)

        prosody_cached = self._get_or_cache(
            audio_path + f"_L{T_h}", "prosody",
            _extract_prosody,
        )

        if isinstance(prosody_cached, tuple) and len(prosody_cached) == 3:
            f0, energy, voiced = prosody_cached
        else:
            # Stale non-tuple cache entry — recompute
            f0_np, energy_np, voiced_np = extract_f0_energy(
                wav_np, self.sr, self.hop_length
            )
            f0     = torch.from_numpy(self._resample_array(f0_np,     T_h))
            energy = torch.from_numpy(self._resample_array(energy_np, T_h))
            voiced = torch.from_numpy(
                self._resample_array(voiced_np.astype(np.float32), T_h) > 0.5
            )


        # --- Optional phoneme labels ---
        # We only include phone_labels when they are genuinely populated
        # (i.e. from a forced-alignment source). When missing from the manifest
        # we set phone_labels=None so collate_fn skips AlignmentLoss entirely,
        # preventing F.cross_entropy from receiving all-padding tensors and
        # returning nan (which would corrupt the total loss).
        phone_labels = None
        if "phone_labels" in sample:
            pl = torch.tensor(sample["phone_labels"][:T_h], dtype=torch.long)
            if len(pl) < T_h:
                pl = F.pad(pl, (0, T_h - len(pl)), value=-100)
            # Only keep if there is at least one valid label (not all -100)
            if (pl != -100).any():
                phone_labels = pl

        return {
            "tokens":       tokens,            # (T_m, 8) int64
            "hubert":       wav2vec2_feats,    # (T_h, 768) float32  [25 Hz GT]
            "f0":           f0,                # (T_h,) float32
            "energy":       energy,            # (T_h,) float32
            "voiced":       voiced,            # (T_h,) bool
            "phone_labels": phone_labels,      # (T_h,) or None
            "audio_path":   audio_path,
        }

    @staticmethod
    def _resample_array(arr: np.ndarray, target_len: int) -> np.ndarray:
        if len(arr) == target_len:
            return arr
        indices = np.linspace(0, len(arr) - 1, target_len)
        return np.interp(indices, np.arange(len(arr)), arr).astype(arr.dtype)


# Backward-compatibility alias — trainer.py imports MimiHuBERTDataset
MimiHuBERTDataset = MimiWav2Vec2Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Collate Function
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[dict]) -> dict:
    """
    Pad variable-length sequences to the longest in the batch.
    Returns a dict with batch tensors and padding masks.
    Feature dim is inferred dynamically from the first sample so this
    works for 768-dim (wav2vec2-base) output.

    Temporal ratio: T_h = T_m * upsample_factor = T_m * 2
    (Mimi 12.5 Hz × 2 = 25 Hz = bridge output 25 Hz = wav2vec2 GT target rate)
    """
    # Sort by descending token length
    batch = sorted(batch, key=lambda x: x["tokens"].shape[0], reverse=True)

    max_T_m = max(b["tokens"].shape[0] for b in batch)
    max_T_h = max_T_m * 2          # upsample_factor = 2 → 25 Hz feature space

    B = len(batch)
    # Infer dims dynamically — supports both 768 (base) and 1024 (large)
    feat_dim       = batch[0]["hubert"].shape[-1]
    num_codebooks  = batch[0]["tokens"].shape[-1]

    tokens_out = torch.zeros(B, max_T_m, num_codebooks, dtype=torch.long)
    hubert_out  = torch.zeros(B, max_T_h, feat_dim)
    f0_out      = torch.zeros(B, max_T_h)
    energy_out  = torch.zeros(B, max_T_h)
    voiced_out  = torch.zeros(B, max_T_h, dtype=torch.bool)
    mask_out    = torch.zeros(B, max_T_h, dtype=torch.bool)   # True = valid
    phone_out   = torch.full((B, max_T_h), -100, dtype=torch.long)

    token_lengths = []
    for i, sample in enumerate(batch):
        T_m = sample["tokens"].shape[0]
        T_h = T_m * 2              # 25 Hz feature space
        tokens_out[i, :T_m]     = sample["tokens"]
        hubert_out[i, :T_h]     = sample["hubert"]
        f0_out[i, :T_h]         = sample["f0"]
        energy_out[i, :T_h]     = sample["energy"]
        voiced_out[i, :T_h]     = sample["voiced"]
        mask_out[i, :T_h]       = True
        if sample["phone_labels"] is not None:
            phone_out[i, :T_h]  = sample["phone_labels"]
        token_lengths.append(T_m)

    token_lengths = torch.tensor(token_lengths, dtype=torch.long)
    frame_lengths = token_lengths * 2   # 25 Hz feature space; used for CTC input_lengths

    return {
        "tokens":         tokens_out,           # (B, T_m, 8)
        "hubert":         hubert_out,           # (B, T_h, 768) float32  [25 Hz GT]
        "f0":             f0_out,               # (B, T_h)
        "energy":         energy_out,           # (B, T_h)
        "voiced_mask":    voiced_out,           # (B, T_h)
        "mask":           mask_out,             # (B, T_h)
        "phone_labels":   phone_out,            # (B, T_h)
        "input_lengths":  frame_lengths,        # (B,) for CTC
        "ctc_targets":    None,                 # populated externally if needed
        "target_lengths": None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader Factories
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict, device: str = "cpu") -> Tuple[DataLoader, DataLoader]:
    train_ds = MimiWav2Vec2Dataset(cfg["data"]["train_manifest"], cfg, "train", device)
    val_ds   = MimiWav2Vec2Dataset(cfg["data"]["val_manifest"],   cfg, "val",   device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )
    return train_loader, val_loader