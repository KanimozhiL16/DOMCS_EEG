"""
losses.py — DOMCS-EEG Combined Loss Functions
================================================
  1. ArcFaceLoss      — identity discrimination on z_id
  2. SupConLoss       — supervised contrastive loss on z_id
  3. StateLoss        — CrossEntropy state classification on z_state
  4. OrthogonalityLoss— cosine(z_id, z_state) ≈ 0
  5. DOMCSLoss        — weighted combination of all four
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LAMBDA_SUPCON, LAMBDA_STATE, LAMBDA_ORTH


# ─── ArcFace Loss ─────────────────────────────────────────────────────────────

class ArcFaceLoss(nn.Module):
    """
    ArcFace logits from ArcFaceHead → CrossEntropyLoss.
    The ArcFaceHead is defined in model.py and computes the margin logits.
    """
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, arc_logits, labels):
        return self.ce(arc_logits, labels)


# ─── Supervised Contrastive Loss ─────────────────────────────────────────────

class SupConLoss(nn.Module):
    """
    Supervised contrastive loss (Khosla et al., NeurIPS 2020).
    Encourages embeddings of same subject to cluster together.

    z: (B, D) L2-normalised embeddings
    labels: (B,) subject IDs
    temperature: τ
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, labels):
        B = z.size(0)
        device = z.device

        # Cosine similarity matrix: (B, B)
        sim = torch.matmul(z, z.T) / self.temperature  # already L2-normalised

        # Mask: 1 if same class AND different sample
        labels = labels.contiguous().view(-1, 1)
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)   # exclude self
        neg_mask = 1 - (labels == labels.T).float()
        neg_mask.fill_diagonal_(0)

        # For numerical stability: subtract max
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Denominator: all other samples (exclude self)
        self_mask = torch.eye(B, device=device).bool()
        exp_sim   = torch.exp(sim)
        exp_sim   = exp_sim.masked_fill(self_mask, 0)
        log_sum   = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Numerator: only positive pairs
        n_pos = pos_mask.sum(dim=1)
        # Avoid division by zero for samples with no positive pair in batch
        valid = n_pos > 0

        loss_per_anchor = -(pos_mask * (sim - log_sum)).sum(dim=1)
        loss_per_anchor = loss_per_anchor / (n_pos + 1e-8)

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return loss_per_anchor[valid].mean()


# ─── State Classification Loss ───────────────────────────────────────────────

class StateLoss(nn.Module):
    """
    CrossEntropyLoss on state_logits vs y_state.
    state_logits: (B, n_state_classes) from StateClassifier(z_state)
    y_state: (B,)  rest=0, task=1
    """
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, state_logits, y_state):
        return self.ce(state_logits, y_state)


# ─── Orthogonality Loss ───────────────────────────────────────────────────────

class OrthogonalityLoss(nn.Module):
    """
    Penalises cosine similarity between z_id and z_state.
    Both are already L2-normalised, so cosine = dot product.
    L_orth = mean(|cosine(z_id, z_state)|)
    """
    def forward(self, z_id, z_state):
        # z_id, z_state: (B, D) already L2-normalised
        cosine = (z_id * z_state).sum(dim=1)   # element-wise dot → (B,)
        return cosine.abs().mean()


# ─── Combined DOMCS Loss ──────────────────────────────────────────────────────

class DOMCSLoss(nn.Module):
    """
    L_total = L_arcface
            + λ_sc    · L_supcon
            + λ_state · L_state
            + λ_orth  · L_orth

    Returns scalar total loss and dict of component losses for logging.
    """
    def __init__(self,
                 lambda_sc    = LAMBDA_SUPCON,
                 lambda_state = LAMBDA_STATE,
                 lambda_orth  = LAMBDA_ORTH):
        super().__init__()
        self.lambda_sc    = lambda_sc
        self.lambda_state = lambda_state
        self.lambda_orth  = lambda_orth

        self.arcface_loss = ArcFaceLoss()
        self.supcon_loss  = SupConLoss(temperature=0.07)
        self.state_loss   = StateLoss()
        self.orth_loss    = OrthogonalityLoss()

    def forward(self, arc_logits, z_id, z_state,
                state_logits, y_subj, y_state):
        """
        arc_logits:   (B, N_subjects)  from ArcFaceHead
        z_id:         (B, 128)         L2-normalised identity embeddings
        z_state:      (B, 128)         L2-normalised state embeddings
        state_logits: (B, n_state)     from StateClassifier
        y_subj:       (B,)             subject labels
        y_state:      (B,)             state labels (0=rest, 1=task)
        """
        L_arc   = self.arcface_loss(arc_logits, y_subj)
        L_sc    = self.supcon_loss(z_id, y_subj)
        L_state = self.state_loss(state_logits, y_state)
        L_orth  = self.orth_loss(z_id, z_state)

        L_total = (L_arc
                   + self.lambda_sc    * L_sc
                   + self.lambda_state * L_state
                   + self.lambda_orth  * L_orth)

        components = {
            "L_arcface": L_arc.item(),
            "L_supcon":  L_sc.item(),
            "L_state":   L_state.item(),
            "L_orth":    L_orth.item(),
            "L_total":   L_total.item(),
        }
        return L_total, components


if __name__ == "__main__":
    # Quick sanity test
    B, D, S, K = 32, 128, 109, 2
    z_id        = F.normalize(torch.randn(B, D), dim=1)
    z_state     = F.normalize(torch.randn(B, D), dim=1)
    arc_logits  = torch.randn(B, S)
    state_logits= torch.randn(B, K)
    y_subj      = torch.randint(0, S, (B,))
    y_state     = torch.randint(0, K, (B,))

    criterion = DOMCSLoss()
    loss, comp = criterion(arc_logits, z_id, z_state, state_logits, y_subj, y_state)
    print(f"Total loss: {loss.item():.4f}")
    for k, v in comp.items():
        print(f"  {k}: {v:.4f}")
    loss.backward()
    print("✓ Backward pass successful")
