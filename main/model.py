"""
model.py — DOMCS-EEG Dual-Space Disentanglement Model
=======================================================
Architecture (exactly as per manuscript):
  INPUT (B, 64, 256)
    └─ CNN encoder [Conv k=7/5/3 | BN | ELU | AdaptiveAvgPool] → f ∈ R^256
         ├─ identity_branch [Linear→LayerNorm→L2-norm] → z_id ∈ R^128
         └─ state_branch [f.detach()→Linear→LayerNorm→L2-norm] → z_state ∈ R^128

Training heads (removed at inference):
  ArcFaceHead: operates on z_id
  StateClassifier: operates on z_state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (N_CHANNELS, WIN_SAMPLES, ENCODER_CHANNELS,
                    ID_DIM, STATE_DIM, N_SUBJECTS, N_STATE_CLASSES,
                    ARC_S, ARC_M)
import math


# ─── CNN Encoder ─────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv1d → BatchNorm → ELU"""
    def __init__(self, in_ch, out_ch, kernel, padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ELU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class EEGEncoder(nn.Module):
    """
    3-layer 1D CNN encoder.
      Conv1: 64→64,  k=7, pad=3
      Conv2: 64→128, k=5, pad=2
      Conv3: 128→256,k=3, pad=1
    → AdaptiveAvgPool1d(1) → f ∈ R^256
    """
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(64,  64,  kernel=7, padding=3)
        self.conv2 = ConvBlock(64,  128, kernel=5, padding=2)
        self.conv3 = ConvBlock(128, 256, kernel=3, padding=1)
        self.pool  = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, 64, 256)
        x = self.conv1(x)   # (B, 64,  256)
        x = self.conv2(x)   # (B, 128, 256)
        x = self.conv3(x)   # (B, 256, 256)
        x = self.pool(x)    # (B, 256, 1)
        return x.squeeze(-1)  # (B, 256)


# ─── Embedding Branches ───────────────────────────────────────────────────────

class IdentityBranch(nn.Module):
    """
    Linear(256→128) → LayerNorm(128) → L2-normalize → z_id
    Receives full gradient from encoder.
    """
    def __init__(self):
        super().__init__()
        self.fc   = nn.Linear(256, ID_DIM, bias=False)
        self.norm = nn.LayerNorm(ID_DIM)

    def forward(self, f):
        z = self.norm(self.fc(f))
        return F.normalize(z, p=2, dim=1)   # L2-normalized z_id


class StateBranch(nn.Module):
    """
    f.detach() → Linear(256→128) → LayerNorm(128) → L2-normalize → z_state
    The .detach() ensures state losses do NOT back-propagate into the encoder.
    """
    def __init__(self):
        super().__init__()
        self.fc   = nn.Linear(256, STATE_DIM, bias=False)
        self.norm = nn.LayerNorm(STATE_DIM)

    def forward(self, f):
        z = self.norm(self.fc(f.detach()))   # CRITICAL: detach from encoder
        return F.normalize(z, p=2, dim=1)   # L2-normalized z_state


# ─── Training Heads ───────────────────────────────────────────────────────────

class ArcFaceHead(nn.Module):
    """
    ArcFace margin loss head.
    Weights are the class prototypes (L2-normalised).
    Combines with CrossEntropyLoss in losses.py.
    """
    def __init__(self, in_dim=ID_DIM, n_classes=N_SUBJECTS,
                 s=ARC_S, m=ARC_M):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, z_id, labels):
        # z_id: (B, 128) L2-normalised
        # weight: (n_classes, 128) — normalise before cosine
        W = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(z_id, W)               # (B, n_classes)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        theta  = torch.acos(cosine)
        # Add margin to the ground-truth class angle
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        target_logits = torch.cos(theta + self.m)
        logits = cosine * (1 - one_hot) + target_logits * one_hot
        return logits * self.s


class StateClassifier(nn.Module):
    """
    Linear classifier on z_state.
    Default: binary (rest=0, task=1). Set n_classes=5 for fine-grained.
    """
    def __init__(self, in_dim=STATE_DIM, n_classes=N_STATE_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, z_state):
        return self.fc(z_state)


# ─── Full Model ───────────────────────────────────────────────────────────────

class DOMCSEEGModel(nn.Module):
    """
    DOMCS-EEG: Disentangled Orthogonal Multi-Constraint State-Invariant
    EEG Biometric Verification.

    forward() returns:
        z_id    — L2-normalised identity embedding (B, 128)
        z_state — L2-normalised state embedding    (B, 128)
        f       — shared encoder feature           (B, 256)
    """
    def __init__(self):
        super().__init__()
        self.encoder       = EEGEncoder()
        self.id_branch     = IdentityBranch()
        self.state_branch  = StateBranch()

    def forward(self, x):
        f       = self.encoder(x)            # (B, 256)
        z_id    = self.id_branch(f)          # (B, 128), grad flows to encoder
        z_state = self.state_branch(f)       # (B, 128), f.detach() inside
        return z_id, z_state, f

    def get_identity_embedding(self, x):
        """Inference-only: returns z_id without state branch."""
        with torch.no_grad():
            f    = self.encoder(x)
            z_id = self.id_branch(f)
        return z_id


# ─── Parameter count ─────────────────────────────────────────────────────────

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    import sys
    model = DOMCSEEGModel()
    total, trainable = count_parameters(model)
    print(f"DOMCSEEGModel — Total params: {total:,}  |  Trainable: {trainable:,}")

    # Smoke test
    x = torch.randn(8, 64, 256)
    z_id, z_state, f = model(x)
    print(f"f      shape: {f.shape}")       # (8, 256)
    print(f"z_id   shape: {z_id.shape}")    # (8, 128)
    print(f"z_state shape: {z_state.shape}")# (8, 128)

    # Verify L2 normalisation
    assert torch.allclose(z_id.norm(dim=1), torch.ones(8), atol=1e-5), "z_id not L2-normalised"
    assert torch.allclose(z_state.norm(dim=1), torch.ones(8), atol=1e-5), "z_state not L2-normalised"
    print("✓ All assertions passed")

    # Verify detach: z_state grad path should NOT reach encoder
    loss = z_state.sum()
    loss.backward()
    enc_grad = model.encoder.conv1.net[0].weight.grad
    assert enc_grad is None or enc_grad.abs().max() < 1e-9, \
        "ERROR: state branch gradient leaking into encoder!"
    print("✓ State branch detach verified — encoder gradient = 0")
