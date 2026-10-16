"""Grayscale DCT U-ViT, extracted from the research implementation.

Parameter names intentionally match legacy checkpoints.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint
import einops

trunc_normal_ = nn.init.trunc_normal_

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        (B, L, C) = x.shape
        qkv = self.qkv(x)
        qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = einops.rearrange(x, 'B H L D -> B L (H D)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip, use_reentrant=False, preserve_rng_state=True)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Expert(nn.Module):
    """ A single expert in a Mixture of Experts layer. Replace the FFN in the Transformer block with this."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TopKRouter(nn.Module):
    """
    Implements a Top-K Gating function for a Mixture of Experts layer.

    This router determines which experts should process each token and calculates
    the weights for combining their outputs. It also computes a load balancing
    loss to encourage even distribution of tokens across experts during training.
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int, noise_eps: float=0.01):
        """
        Args:
            d_model (int): The hidden dimension of the input tokens.
            num_experts (int): The total number of experts available.
            top_k (int): The number of experts to route each token to.
            noise_eps (float): The scaling factor for the noise added during training.
        """
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_eps = noise_eps
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        A more robust implementation of the Top-K Gating function with STE.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            tuple: A tuple containing:
                - final_weights (torch.Tensor): The weights for the experts, of shape [batch_size, seq_len, num_experts].
                - expert_indices (torch.Tensor): The indices of the selected experts, of shape [batch_size, seq_len, top_k].
                - aux_loss (torch.Tensor): The auxiliary load balancing loss.
        """
        (batch_size, seq_len, _) = x.shape
        x_flat = x.view(-1, self.d_model)
        num_tokens = x_flat.shape[0]
        logits = self.gate(x_flat)
        if self.training and self.noise_eps > 0:
            noise = torch.randn_like(logits) * self.noise_eps
            logits += noise
        logits = torch.clamp(logits, min=-100.0, max=100.0)
        logits_max = torch.max(logits, dim=-1, keepdim=True)[0].detach()
        logits = logits - logits_max
        router_probs = F.softmax(logits.float(), dim=-1)
        (topk_probs, topk_indices) = torch.topk(router_probs, self.top_k, dim=-1)
        hard_mask = F.one_hot(topk_indices, num_classes=self.num_experts).sum(dim=1)
        hard_mask = hard_mask.float()
        ste_weights = (hard_mask - router_probs).detach() + router_probs
        f_i = ste_weights.sum(0) / num_tokens
        P_i = router_probs.sum(0) / num_tokens
        f_i = torch.clamp(f_i, min=1e-06)
        P_i = torch.clamp(P_i, min=1e-06)
        aux_loss = self.num_experts * torch.sum(f_i * P_i)
        aux_loss = torch.clamp(aux_loss, max=100.0)
        # Equation 5: retain the selected softmax probabilities without top-k
        # renormalization. Keep the STE mask for selection gradients.
        final_weights = ste_weights * router_probs
        return (final_weights.view(batch_size, seq_len, -1), topk_indices.view(batch_size, seq_len, -1), aux_loss)

class Block_MoE(nn.Module):
    """Sparse FFN with probability-weighted top-k straight-through routing."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False,
                 use_checkpoint=False, num_experts=4, top_k=2, noise_eps=1e-2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale)
        self.norm2 = norm_layer(dim)
        self.router = TopKRouter(dim, num_experts, top_k, noise_eps)
        self.experts = nn.ModuleList(Expert(dim, int(dim * mlp_ratio), act_layer=act_layer)
                                     for _ in range(num_experts))
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, skip, use_reentrant=False, preserve_rng_state=True)
        return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        normalized = self.norm2(x)
        weights, indices, aux_loss = self.router(normalized)
        flat_x = normalized.reshape(-1, normalized.shape[-1])
        flat_weights = weights.reshape(-1, len(self.experts))
        flat_indices = indices.reshape(-1, indices.shape[-1])
        output = torch.zeros_like(flat_x)
        for i, expert in enumerate(self.experts):
            selected = torch.where(flat_indices == i)[0]
            if selected.numel():
                values = expert(flat_x[selected]).clamp(-1e6, 1e6)
                values = values * flat_weights[selected, i, None]
                output.index_add_(0, selected, values.to(output.dtype))
        return x + output.reshape_as(normalized), aux_loss

class UViT_greyscale_MoE(nn.Module):
    """Unconditional grayscale U-ViT with first/last MoE blocks.

    As in the original code, depth counts the paired encoder/decoder blocks;
    there is one additional dense middle block. State-dict names are preserved.
    """
    def __init__(self, tokens, low_freqs, embed_dim=768, depth=16, num_heads=12,
                 mlp_ratio=4., qkv_bias=False, mlp_time_embed=True,
                 use_checkpoint=False, MoE=None):
        super().__init__()
        moe = dict(depth=1, num_experts=4, router='topk', top_k=2, noise_eps=.01)
        if MoE is not None:
            moe.update(MoE)
        if depth < 2 or depth % 2 or embed_dim < 2 or embed_dim % num_heads:
            raise ValueError('depth must be positive and even; embed_dim must be divisible by num_heads')
        if tokens < 1 or low_freqs < 1:
            raise ValueError('tokens and low_freqs must be positive')
        if moe['depth'] != 1 or moe['router'] != 'topk':
            raise ValueError('Only first/last top-k MoE blocks are included in this release')
        if not 1 <= moe['top_k'] <= moe['num_experts']:
            raise ValueError('top_k must be between 1 and num_experts')
        self.num_features = self.embed_dim = embed_dim
        self.tokens = tokens
        self.DCT_coes = low_freqs
        self.extras = 1
        self.proj = nn.Linear(low_freqs * 4, embed_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim)) if mlp_time_embed else nn.Identity()
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + tokens, embed_dim))
        common = dict(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, use_checkpoint=use_checkpoint)
        sparse = dict(num_experts=moe['num_experts'], top_k=moe['top_k'], noise_eps=moe['noise_eps'])
        self.in_blocks = nn.ModuleList([
            Block_MoE(**common, **sparse) if i == 0 else Block(**common)
            for i in range(depth // 2)])
        self.mid_block = Block(**common)
        self.out_blocks = nn.ModuleList([
            Block_MoE(**common, skip=True, **sparse) if i == depth // 2 - 1
            else Block(**common, skip=True) for i in range(depth // 2)])
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, low_freqs * 4)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x, timesteps):
        if x.ndim != 3 or x.shape[1:] != (self.tokens, 4 * self.DCT_coes):
            raise ValueError(f'Expected [batch, {self.tokens}, {4 * self.DCT_coes}] DCT input')
        if timesteps.shape != (x.shape[0],):
            raise ValueError('timesteps must have shape [batch]')
        x = self.proj(x)
        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim)).unsqueeze(1)
        x = torch.cat((time_token, x), dim=1) + self.pos_embed
        skips, auxiliary = [], []
        for block in self.in_blocks:
            if isinstance(block, Block_MoE):
                x, aux = block(x)
                auxiliary.append(aux)
            else:
                x = block(x)
            skips.append(x)
        x = self.mid_block(x)
        for block in self.out_blocks:
            if isinstance(block, Block_MoE):
                x, aux = block(x, skips.pop())
                auxiliary.append(aux)
            else:
                x = block(x, skips.pop())
        prediction = self.decoder_pred(self.norm(x)[:, 1:, :])
        return prediction, torch.stack(auxiliary).mean()
