"""
losses.py — Multi-Loss Training Objectives
===========================================
Implements all loss components for Mimi-to-Wav2Vec2 bridge training:
  - Reconstruction (MSE + cosine)
  - CTC / ASR consistency
  - Prosody (pitch + energy)
  - Adversarial (GAN)
  - Statistical (mean/var)
  - Temporal smoothness
  - Alignment (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


# ──────────────────────────────────────────────────────────────────────────────
# 1. Reconstruction Loss
# ──────────────────────────────────────────────────────────────────────────────

class ReconstructionLoss(nn.Module):
    """Primary MSE + cosine loss against Wav2Vec2 target features."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        # Also include cosine similarity term for directional alignment
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        pred, target: (B, T, output_dim)
        mask:         (B, T) bool, True = valid frame
        """
        if mask is not None:
            # Apply mask
            pred_m   = pred[mask]
            target_m = target[mask]
        else:
            pred_m, target_m = pred, target

        mse_loss = self.mse(pred_m, target_m)
        cos_sim  = self.cos(pred_m.view(-1, pred_m.size(-1)),
                            target_m.view(-1, target_m.size(-1))).mean()
        # We want cos_sim → 1
        cos_loss = 1.0 - cos_sim

        total = mse_loss + 0.1 * cos_loss
        return total, {"recon_mse": mse_loss.item(), "recon_cos": cos_loss.item()}


# ──────────────────────────────────────────────────────────────────────────────
# 2. CTC Loss  (ASR Consistency)
# ──────────────────────────────────────────────────────────────────────────────

class CTCConsistencyLoss(nn.Module):
    """
    Frozen ASR head on top of predicted features.
    Forces the bridge output to remain phonetically intelligible.
    """

    def __init__(self, input_dim: int = 768, vocab_size: int = 31):
        super().__init__()
        self.ctc_head = nn.Linear(input_dim, vocab_size)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
        # Head stays frozen after loading pretrained weights
        self._frozen = False

    def freeze(self):
        for p in self.ctc_head.parameters():
            p.requires_grad = False
        self._frozen = True

    def load_pretrained(self, state_dict: dict):
        self.ctc_head.load_state_dict(state_dict)
        self.freeze()

    def forward(
        self,
        pred: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        pred:           (B, T, input_dim)
        targets:        (sum(target_lengths),)  flat phoneme indices
        input_lengths:  (B,) frame counts (pre-padding)
        target_lengths: (B,) target lengths
        """
        # CTC requires float32 log-probs; cast from half if needed
        logits = self.ctc_head(pred.float())
        log_probs = F.log_softmax(logits, dim=-1)   # (B, T, V)
        log_probs = log_probs.transpose(0, 1)       # (T, B, V) for CTCLoss

        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        return loss, {"ctc": loss.item()}


# ──────────────────────────────────────────────────────────────────────────────
# 3. Prosody Loss
# ──────────────────────────────────────────────────────────────────────────────

class ProsodyLoss(nn.Module):
    """
    Predict pitch (F0) and energy from bridge output features.
    Compare against ground-truth extracted from audio.

    dtype safety: ground-truth tensors (float32 from dataloader) are cast to
    match the prediction dtype so this is safe under torch.amp.autocast.
    """

    def __init__(self, input_dim: int = 768):
        super().__init__()
        # Small heads to predict prosodic quantities
        self.f0_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.energy_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        pred: torch.Tensor,
        f0_gt: torch.Tensor,
        energy_gt: torch.Tensor,
        voiced_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        pred:       (B, T, input_dim)  — may be float16 under autocast
        f0_gt:      (B, T)  normalised log-F0, 0 for unvoiced   [float32]
        energy_gt:  (B, T)  normalised log-energy               [float32]
        voiced_mask:(B, T)  bool — only compute F0 on voiced frames
        """
        f0_pred     = self.f0_head(pred).squeeze(-1)      # (B, T)
        energy_pred = self.energy_head(pred).squeeze(-1)  # (B, T)

        # ── dtype alignment ────────────────────────────────────────────────────
        # pred (and therefore f0_pred / energy_pred) may be float16 under AMP.
        # Ground-truth tensors from the dataloader are always float32.
        # Cast gt → pred dtype to avoid "Half vs Float" runtime errors.
        target_dtype = f0_pred.dtype
        f0_gt     = f0_gt.to(target_dtype)
        energy_gt = energy_gt.to(target_dtype)
        # ──────────────────────────────────────────────────────────────────────

        energy_loss = F.mse_loss(energy_pred, energy_gt)

        if voiced_mask is not None and voiced_mask.any():
            f0_loss = F.mse_loss(f0_pred[voiced_mask], f0_gt[voiced_mask])
        else:
            f0_loss = F.mse_loss(f0_pred, f0_gt)

        total = f0_loss + energy_loss
        return total, {"prosody_f0": f0_loss.item(), "prosody_energy": energy_loss.item()}


# ──────────────────────────────────────────────────────────────────────────────
# 4. Adversarial Loss
# ──────────────────────────────────────────────────────────────────────────────

class AdversarialLoss(nn.Module):
    """
    Hinge GAN loss for feature distribution matching.
    Used to train both generator (bridge) and discriminator.
    """

    def __init__(self, loss_type: str = "hinge"):
        super().__init__()
        assert loss_type in ("hinge", "bce", "wgan")
        self.loss_type = loss_type

    def discriminator_loss(
        self, real_logits: torch.Tensor, fake_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        if self.loss_type == "hinge":
            d_real = F.relu(1.0 - real_logits).mean()
            d_fake = F.relu(1.0 + fake_logits).mean()
            loss   = d_real + d_fake
        elif self.loss_type == "bce":
            ones  = torch.ones_like(real_logits)
            zeros = torch.zeros_like(fake_logits)
            loss  = (F.binary_cross_entropy_with_logits(real_logits, ones)
                     + F.binary_cross_entropy_with_logits(fake_logits, zeros))
        elif self.loss_type == "wgan":
            loss = -real_logits.mean() + fake_logits.mean()
        return loss, {"d_loss": loss.item()}

    def generator_loss(self, fake_logits: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        if self.loss_type == "hinge":
            loss = -fake_logits.mean()
        elif self.loss_type == "bce":
            ones = torch.ones_like(fake_logits)
            loss = F.binary_cross_entropy_with_logits(fake_logits, ones)
        elif self.loss_type == "wgan":
            loss = -fake_logits.mean()
        return loss, {"g_adv_loss": loss.item()}

    def forward(self, fake_logits):
        """Alias for generator loss (called in bridge training step)."""
        return self.generator_loss(fake_logits)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Feature Statistics Loss
# ──────────────────────────────────────────────────────────────────────────────

class StatisticsLoss(nn.Module):
    """
    Match first and second moments of predicted vs real feature distributions.
    Encourages the bridge to stay in the HuBERT feature manifold.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        pred, target: (B, T, D)
        mask:         (B, T) bool
        """
        if mask is not None:
            pred_flat   = pred[mask]    # (N, D)
            target_flat = target[mask]
        else:
            pred_flat   = pred.reshape(-1, pred.size(-1))
            target_flat = target.reshape(-1, target.size(-1))

        mu_pred, mu_tgt   = pred_flat.mean(0), target_flat.mean(0)
        std_pred, std_tgt = pred_flat.std(0) + 1e-6, target_flat.std(0) + 1e-6

        mean_loss = F.mse_loss(mu_pred, mu_tgt)
        std_loss  = F.mse_loss(std_pred, std_tgt)

        # Optional: channel-wise correlation
        total = mean_loss + std_loss
        return total, {"stat_mean": mean_loss.item(), "stat_std": std_loss.item()}


# ──────────────────────────────────────────────────────────────────────────────
# 6. Temporal Smoothness Loss
# ──────────────────────────────────────────────────────────────────────────────

class SmoothnessLoss(nn.Module):
    """
    Penalise abrupt first-order discontinuities in predicted features.
    L_smooth = mean || x_t - x_{t-1} ||^2
    """

    def forward(
        self,
        pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """pred: (B, T, D)"""
        diff = pred[:, 1:, :] - pred[:, :-1, :]   # (B, T-1, D)

        if mask is not None:
            # Only compute between consecutive valid frames
            valid = mask[:, 1:] & mask[:, :-1]     # (B, T-1)
            diff  = diff[valid]

        loss = (diff ** 2).mean()
        return loss, {"smooth": loss.item()}


# ──────────────────────────────────────────────────────────────────────────────
# 7. Alignment Loss (optional)
# ──────────────────────────────────────────────────────────────────────────────

class AlignmentLoss(nn.Module):
    """
    Frame-level phoneme cross-entropy using forced-alignment labels.
    Encourages temporal precision in phoneme boundaries.
    """

    def __init__(self, input_dim: int = 768, num_phones: int = 40):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_phones)

    def forward(
        self,
        pred: torch.Tensor,
        phone_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        pred:         (B, T, output_dim)
        phone_labels: (B, T) long — phoneme index, -1 = ignore
        mask:         (B, T) bool
        """
        logits = self.classifier(pred)            # (B, T, num_phones)

        if mask is not None:
            phone_labels = phone_labels.masked_fill(~mask, -100)

        # Guard: if every label is the ignore index (-100), cross_entropy returns
        # nan (0/0). Return a zero loss instead so total is not contaminated.
        valid = (phone_labels != -100)
        if not valid.any():
            zero = pred.new_zeros(1).squeeze()
            return zero, {"alignment": 0.0}

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            phone_labels.view(-1),
            ignore_index=-100,
        )
        return loss, {"alignment": loss.item()}


# ──────────────────────────────────────────────────────────────────────────────
# Combined Loss Manager
# ──────────────────────────────────────────────────────────────────────────────

class BridgeLoss(nn.Module):
    """
    Aggregates all losses with configurable weights.
    All sub-modules that operate on bridge output features are constructed
    with the correct output_dim (768 for Wav2Vec2-base-960h) read from cfg.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        w = cfg["training"]["loss_weights"]
        self.weights = w

        # Feature dimension of the bridge output (must match model.output_dim)
        feat_dim = cfg["model"]["output_dim"]

        self.recon     = ReconstructionLoss()
        self.ctc       = CTCConsistencyLoss(
            input_dim=feat_dim,
            vocab_size=cfg["training"]["ctc_vocab_size"],
        )
        self.prosody   = ProsodyLoss(input_dim=feat_dim)
        self.adv       = AdversarialLoss()
        self.stat      = StatisticsLoss()
        self.smooth    = SmoothnessLoss()
        self.alignment = AlignmentLoss(
            input_dim=feat_dim,
            num_phones=cfg["training"].get("num_phones", 40),
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        batch: dict,
        fake_disc_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        pred:             (B, T, output_dim)
        target:           (B, T, output_dim)  HuBERT features
        batch:            dict with optional keys: mask, f0, energy, voiced_mask,
                          ctc_targets, input_lengths, target_lengths, phone_labels
        fake_disc_logits: discriminator output on predicted features (for gen loss)
        """
        mask = batch.get("mask", None)
        logs = {}
        total = torch.tensor(0.0, device=pred.device)

        # 1. Reconstruction
        if self.weights.get("recon", 0) > 0:
            l, d = self.recon(pred, target, mask)
            total = total + self.weights["recon"] * l
            logs.update(d)

        # 2. CTC
        if (self.weights.get("ctc", 0) > 0
                and "ctc_targets" in batch
                and batch["ctc_targets"] is not None):
            l, d = self.ctc(
                pred,
                batch["ctc_targets"],
                batch["input_lengths"],
                batch["target_lengths"],
            )
            total = total + self.weights["ctc"] * l
            logs.update(d)

        # 3. Prosody
        if (self.weights.get("prosody", 0) > 0
                and "f0" in batch and batch["f0"] is not None):
            l, d = self.prosody(
                pred, batch["f0"], batch["energy"], batch.get("voiced_mask")
            )
            total = total + self.weights["prosody"] * l
            logs.update(d)

        # 4. Adversarial (generator side)
        if (self.weights.get("adv", 0) > 0
                and fake_disc_logits is not None):
            l, d = self.adv.generator_loss(fake_disc_logits)
            total = total + self.weights["adv"] * l
            logs.update(d)

        # 5. Statistics
        if self.weights.get("stat", 0) > 0:
            l, d = self.stat(pred, target, mask)
            total = total + self.weights["stat"] * l
            logs.update(d)

        # 6. Smoothness
        if self.weights.get("smooth", 0) > 0:
            l, d = self.smooth(pred, mask)
            total = total + self.weights["smooth"] * l
            logs.update(d)

        # 7. Alignment
        if (self.weights.get("alignment", 0) > 0
                and "phone_labels" in batch
                and batch["phone_labels"] is not None):
            l, d = self.alignment(pred, batch["phone_labels"], mask)
            total = total + self.weights["alignment"] * l
            logs.update(d)

        logs["total"] = total.item()
        return total, logs
