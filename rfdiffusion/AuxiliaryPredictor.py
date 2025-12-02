"""
Auxiliary prediction networks for protein structure properties.

This module contains neural network classes for predicting various structural
and quality properties from learned feature representations:
- Distance and orientation prediction between residues
- Masked token prediction for sequence recovery
- LDDT (Local Distance Difference Test) for structure quality
- Experimental resolution prediction

These networks are used as auxiliary heads during training to provide
additional supervision signals and can be used for quality assessment
during inference.
"""
import torch
import torch.nn as nn

class DistanceNetwork(nn.Module):
    """
    Network for predicting inter-residue distances and orientations.

    Predicts discretized values for:
    - Distance between C-beta atoms (37 bins)
    - Omega: dihedral CA-CB-CB-CA (37 bins)
    - Theta: dihedral N-CA-CB-CB (37 bins)
    - Phi: planar angle CA-CB-CB (19 bins)

    The distance and omega predictions are symmetrized across the pair dimensions,
    while theta and phi are asymmetric.
    """
    def __init__(self, n_feat, p_drop=0.1):
        super(DistanceNetwork, self).__init__()
        #
        self.proj_symm = nn.Linear(n_feat, 37*2)
        self.proj_asymm = nn.Linear(n_feat, 37+19)
    
        self.reset_parameter()
    
    def reset_parameter(self):
        # initialize linear layer for final logit prediction
        nn.init.zeros_(self.proj_symm.weight)
        nn.init.zeros_(self.proj_asymm.weight)
        nn.init.zeros_(self.proj_symm.bias)
        nn.init.zeros_(self.proj_asymm.bias)

    def forward(self, x):
        """
        Predict distance and orientation logits from pair features.

        Args:
            x (torch.Tensor): Pair features of shape (B, L, L, C)

        Returns:
            tuple: (logits_dist, logits_omega, logits_theta, logits_phi)
                All of shape (B, n_bins, L, L)
        """
        # input: pair info (B, L, L, C)

        # predict theta, phi (non-symmetric)
        logits_asymm = self.proj_asymm(x)
        logits_theta = logits_asymm[:,:,:,:37].permute(0,3,1,2)
        logits_phi = logits_asymm[:,:,:,37:].permute(0,3,1,2)

        # predict dist, omega
        logits_symm = self.proj_symm(x)
        logits_symm = logits_symm + logits_symm.permute(0,2,1,3)
        logits_dist = logits_symm[:,:,:,:37].permute(0,3,1,2)
        logits_omega = logits_symm[:,:,:,37:].permute(0,3,1,2)

        return logits_dist, logits_omega, logits_theta, logits_phi

class MaskedTokenNetwork(nn.Module):
    """
    Network for predicting masked amino acid identities.

    Used for sequence recovery tasks and as an auxiliary training objective.
    Predicts amino acid type for each position from MSA features.
    """
    def __init__(self, n_feat):
        super(MaskedTokenNetwork, self).__init__()
        self.proj = nn.Linear(n_feat, 21)
        
        self.reset_parameter()
    
    def reset_parameter(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        """
        Predict amino acid type logits from MSA features.

        Args:
            x (torch.Tensor): MSA features of shape (B, N, L, n_feat)

        Returns:
            torch.Tensor: Logits of shape (B, 21, N*L) for 21 amino acid types
        """
        B, N, L = x.shape[:3]
        logits = self.proj(x).permute(0,3,1,2).reshape(B, -1, N*L)

        return logits

class LDDTNetwork(nn.Module):
    """
    Network for predicting per-residue LDDT (Local Distance Difference Test) scores.

    LDDT is a measure of local structure quality that compares predicted and
    reference structures. Higher scores (0-1) indicate better local accuracy.
    The output is discretized into bins for classification.
    """
    def __init__(self, n_feat, n_bin_lddt=50):
        super(LDDTNetwork, self).__init__()
        self.proj = nn.Linear(n_feat, n_bin_lddt)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        """
        Predict LDDT logits from per-residue features.

        Args:
            x (torch.Tensor): Per-residue features of shape (B, L, n_feat)

        Returns:
            torch.Tensor: LDDT logits of shape (B, n_bin_lddt, L)
        """
        logits = self.proj(x) # (B, L, 50)

        return logits.permute(0,2,1)

class ExpResolvedNetwork(nn.Module):
    """
    Network for predicting experimental resolution or quality scores.

    Predicts a single scalar value per structure that indicates the expected
    experimental resolution or overall quality of the structure.
    """
    def __init__(self, d_msa, d_state, p_drop=0.1):
        super(ExpResolvedNetwork, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_state = nn.LayerNorm(d_state)
        self.proj = nn.Linear(d_msa+d_state, 1)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, seq, state):
        """
        Predict experimental resolution from sequence and state features.

        Args:
            seq (torch.Tensor): Sequence features of shape (B, L, d_msa)
            state (torch.Tensor): State features of shape (B, L, d_state)

        Returns:
            torch.Tensor: Resolution predictions of shape (B, L)
        """
        B, L = seq.shape[:2]

        seq = self.norm_msa(seq)
        state = self.norm_state(state)
        feat = torch.cat((seq, state), dim=-1)
        logits = self.proj(feat)
        return logits.reshape(B, L)



