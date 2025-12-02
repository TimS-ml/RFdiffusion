"""
Attention mechanisms and neural network building blocks.

This module implements various attention mechanisms used in the RFdiffusion model:
- Standard multi-head attention
- Attention with learned bias (for incorporating pair/structure information)
- MSA row and column attention (AlphaFold2-style)
- Tied axial attention for efficient 2D feature processing
- Feed-forward layers with residual connections

These components are the core building blocks of the three-track architecture
(MSA track, Pair track, Structure track) that processes protein information.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from opt_einsum import contract as einsum
from rfdiffusion.util_module import init_lecun_normal

class FeedForwardLayer(nn.Module):
    """
    Position-wise feed-forward network with residual connection.

    Applies two linear transformations with ReLU activation in between:
    FFN(x) = Linear2(Dropout(ReLU(Linear1(LayerNorm(x)))))

    This is used after attention layers to add non-linearity and increase
    model capacity.
    """
    def __init__(self, d_model, r_ff, p_drop=0.1):
        super(FeedForwardLayer, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model*r_ff)
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_model*r_ff, d_model)

        self.reset_parameter()

    def reset_parameter(self):
        # initialize linear layer right before ReLu: He initializer (kaiming normal)
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear1.bias)

        # initialize linear layer right before residual connection: zero initialize
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, src):
        src = self.norm(src)
        src = self.linear2(self.dropout(F.relu_(self.linear1(src))))
        return src

class Attention(nn.Module):
    """
    Standard multi-head attention mechanism.

    Implements the scaled dot-product attention:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V

    Uses separate query, key, and value projections for each head, then
    combines the outputs with a final linear projection.
    """
    def __init__(self, d_query, d_key, n_head, d_hidden, d_out):
        super(Attention, self).__init__()
        self.h = n_head
        self.dim = d_hidden
        #
        self.to_q = nn.Linear(d_query, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_key, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_key, n_head*d_hidden, bias=False)
        #
        self.to_out = nn.Linear(n_head*d_hidden, d_out)
        self.scaling = 1/math.sqrt(d_hidden)
        #
        # initialize all parameters properly
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, query, key, value):
        B, Q = query.shape[:2]
        B, K = key.shape[:2]
        #
        query = self.to_q(query).reshape(B, Q, self.h, self.dim)
        key = self.to_k(key).reshape(B, K, self.h, self.dim)
        value = self.to_v(value).reshape(B, K, self.h, self.dim)
        #
        query = query * self.scaling
        attn = einsum('bqhd,bkhd->bhqk', query, key)
        attn = F.softmax(attn, dim=-1)
        #
        out = einsum('bhqk,bkhd->bqhd', attn, value)
        out = out.reshape(B, Q, self.h*self.dim)
        #
        out = self.to_out(out)

        return out

class AttentionWithBias(nn.Module):
    """
    Multi-head attention with learned bias and gating.

    Extends standard attention with:
    1. Learned bias added to attention logits (for incorporating pair features)
    2. Gating mechanism to control information flow
    3. Layer normalization on both input and bias

    The bias allows the attention to be conditioned on pairwise information,
    which is crucial for protein structure modeling.
    """
    def __init__(self, d_in=256, d_bias=128, n_head=8, d_hidden=32):
        super(AttentionWithBias, self).__init__()
        self.norm_in = nn.LayerNorm(d_in)
        self.norm_bias = nn.LayerNorm(d_bias)
        #
        self.to_q = nn.Linear(d_in, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_in, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_in, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_bias, n_head, bias=False)
        self.to_g = nn.Linear(d_in, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_in)

        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        
        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x, bias):
        B, L = x.shape[:2]
        #
        x = self.norm_in(x)
        bias = self.norm_bias(bias)
        #
        query = self.to_q(x).reshape(B, L, self.h, self.dim)
        key = self.to_k(x).reshape(B, L, self.h, self.dim)
        value = self.to_v(x).reshape(B, L, self.h, self.dim)
        bias = self.to_b(bias) # (B, L, L, h)
        gate = torch.sigmoid(self.to_g(x))
        #
        key = key * self.scaling
        attn = einsum('bqhd,bkhd->bqkh', query, key)
        attn = attn + bias
        attn = F.softmax(attn, dim=-2)
        #
        out = einsum('bqkh,bkhd->bqhd', attn, value).reshape(B, L, -1)
        out = gate * out
        #
        out = self.to_out(out)
        return out

# MSA Attention (row/column) inspired by AlphaFold architecture
class SequenceWeight(nn.Module):
    """
    Compute sequence-wise attention weights for MSA processing.

    Computes attention weights that determine how much each MSA sequence
    contributes to the final representation. Uses the query sequence (first
    sequence in MSA) to weight all sequences.
    """
    def __init__(self, d_msa, n_head, d_hidden, p_drop=0.1):
        super(SequenceWeight, self).__init__()
        self.h = n_head
        self.dim = d_hidden
        self.scale = 1.0 / math.sqrt(self.dim)

        self.to_query = nn.Linear(d_msa, n_head*d_hidden)
        self.to_key = nn.Linear(d_msa, n_head*d_hidden)
        self.dropout = nn.Dropout(p_drop)

        self.reset_parameter()
    
    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_query.weight)
        nn.init.xavier_uniform_(self.to_key.weight)

    def forward(self, msa):
        B, N, L = msa.shape[:3]
       
        tar_seq = msa[:,0]
        
        q = self.to_query(tar_seq).view(B, 1, L, self.h, self.dim)
        k = self.to_key(msa).view(B, N, L, self.h, self.dim)
        
        q = q * self.scale
        attn = einsum('bqihd,bkihd->bkihq', q, k)
        attn = F.softmax(attn, dim=1)
        return self.dropout(attn)

class MSARowAttentionWithBias(nn.Module):
    """
    Row-wise attention over MSA sequences with pair bias.

    For each residue position, attends over all MSA sequences at that position.
    The attention is biased by pairwise information (from the pair track) to
    incorporate structural context. Also uses sequence weighting to handle
    varying MSA depths.

    This is a key component of the MSA track that allows information exchange
    between different sequences in the alignment.
    """
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_hidden=32):
        super(MSARowAttentionWithBias, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        #
        self.seq_weight = SequenceWeight(d_msa, n_head, d_hidden, p_drop=0.1)
        self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_pair, n_head, bias=False)
        self.to_g = nn.Linear(d_msa, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_msa)

        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden

        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        
        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa, pair): # TODO: make this as tied-attention
        B, N, L = msa.shape[:3]
        #
        msa = self.norm_msa(msa)
        pair = self.norm_pair(pair)
        #
        seq_weight = self.seq_weight(msa) # (B, N, L, h, 1)
        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        key = self.to_k(msa).reshape(B, N, L, self.h, self.dim)
        value = self.to_v(msa).reshape(B, N, L, self.h, self.dim)
        bias = self.to_b(pair) # (B, L, L, h)
        gate = torch.sigmoid(self.to_g(msa))
        #
        query = query * seq_weight.expand(-1, -1, -1, -1, self.dim)
        key = key * self.scaling
        attn = einsum('bsqhd,bskhd->bqkh', query, key)
        attn = attn + bias
        attn = F.softmax(attn, dim=-2)
        #
        out = einsum('bqkh,bskhd->bsqhd', attn, value).reshape(B, N, L, -1)
        out = gate * out
        #
        out = self.to_out(out)
        return out

class MSAColAttention(nn.Module):
    """
    Column-wise attention over MSA residue positions.

    For each MSA sequence, attends over all residue positions. This allows
    information to flow along the sequence dimension, capturing long-range
    dependencies within each aligned sequence.

    Complementary to MSARowAttention: row attention mixes sequences, column
    attention mixes positions.
    """
    def __init__(self, d_msa=256, n_head=8, d_hidden=32):
        super(MSAColAttention, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        #
        self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_g = nn.Linear(d_msa, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_msa)

        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa):
        B, N, L = msa.shape[:3]
        #
        msa = self.norm_msa(msa)
        #
        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        key = self.to_k(msa).reshape(B, N, L, self.h, self.dim)
        value = self.to_v(msa).reshape(B, N, L, self.h, self.dim)
        gate = torch.sigmoid(self.to_g(msa))
        #
        query = query * self.scaling
        attn = einsum('bqihd,bkihd->bihqk', query, key)
        attn = F.softmax(attn, dim=-1)
        #
        out = einsum('bihqk,bkihd->bqihd', attn, value).reshape(B, N, L, -1)
        out = gate * out
        #
        out = self.to_out(out)
        return out

class MSAColGlobalAttention(nn.Module):
    """
    Global column attention with shared query across all sequences.

    A memory-efficient variant of MSAColAttention that uses a single global
    query (averaged over sequences) instead of per-sequence queries. This
    reduces memory usage for large MSAs while maintaining the ability to
    capture long-range position dependencies.
    """
    def __init__(self, d_msa=64, n_head=8, d_hidden=8):
        super(MSAColGlobalAttention, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        #
        self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, d_hidden, bias=False)
        self.to_g = nn.Linear(d_msa, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_msa)

        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa):
        B, N, L = msa.shape[:3]
        #
        msa = self.norm_msa(msa)
        #
        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        query = query.mean(dim=1) # (B, L, h, dim)
        key = self.to_k(msa) # (B, N, L, dim)
        value = self.to_v(msa) # (B, N, L, dim)
        gate = torch.sigmoid(self.to_g(msa)) # (B, N, L, h*dim)
        #
        query = query * self.scaling
        attn = einsum('bihd,bkid->bihk', query, key) # (B, L, h, N)
        attn = F.softmax(attn, dim=-1)
        #
        out = einsum('bihk,bkid->bihd', attn, value).reshape(B, 1, L, -1) # (B, 1, L, h*dim)
        out = gate * out # (B, N, L, h*dim)
        #
        out = self.to_out(out)
        return out

class BiasedAxialAttention(nn.Module):
    """
    Tied axial attention with structural bias for 2D pair features.

    Processes 2D pair representations (L x L) using tied attention along one axis
    (either row or column). "Tied" means the same attention weights are shared
    across the other axis, which is more parameter-efficient than full 2D attention.

    The attention is biased by structural information (e.g., distances) to
    incorporate geometric constraints. This is used in the Pair track to update
    pairwise residue features.
    """
    def __init__(self, d_pair, d_bias, n_head, d_hidden, p_drop=0.1, is_row=True):
        super(BiasedAxialAttention, self).__init__()
        #
        self.is_row = is_row
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_bias = nn.LayerNorm(d_bias)

        self.to_q = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_bias, n_head, bias=False) 
        self.to_g = nn.Linear(d_pair, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_pair)
        
        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        
        # initialize all parameters properly
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, pair, bias):
        # pair: (B, L, L, d_pair)
        B, L = pair.shape[:2]
        
        if self.is_row:
            pair = pair.permute(0,2,1,3)
            bias = bias.permute(0,2,1,3)

        pair = self.norm_pair(pair)
        bias = self.norm_bias(bias)
        
        query = self.to_q(pair).reshape(B, L, L, self.h, self.dim)
        key = self.to_k(pair).reshape(B, L, L, self.h, self.dim)
        value = self.to_v(pair).reshape(B, L, L, self.h, self.dim)
        bias = self.to_b(bias) # (B, L, L, h)
        gate = torch.sigmoid(self.to_g(pair)) # (B, L, L, h*dim) 
        
        query = query * self.scaling
        key = key / math.sqrt(L) # normalize for tied attention
        attn = einsum('bnihk,bnjhk->bijh', query, key) # tied attention
        attn = attn + bias # apply bias
        attn = F.softmax(attn, dim=-2) # (B, L, L, h)
        
        out = einsum('bijh,bkjhd->bikhd', attn, value).reshape(B, L, L, -1)
        out = gate * out
        
        out = self.to_out(out)
        if self.is_row:
            out = out.permute(0,2,1,3)
        return out

