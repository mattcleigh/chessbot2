"""Some classes to describe transformer architectures."""

import logging
from dataclasses import dataclass

import torch as T
import torch.nn.functional as F
from torch import nn

log = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    dim: int = 128
    vocab_size: int = 20
    seq_len: int = 65
    class_tokens: int = 1
    output_dim: int = 3
    num_layers: int = 8
    num_heads: int = 8
    ff_mult: int = 2


def norm(x):
    """Simple rmsnorm with no learnable parameters."""
    return F.rms_norm(x, (x.size(-1),))  # note that this will run in bf16, seems ok


class SwiGLUNet(nn.Module):
    """Simple gated bilinear feedfoward network with the SiLU activation."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.ff_mult * config.dim)  # Seperate for Mu
        self.w2 = nn.Linear(config.dim, config.ff_mult * config.dim)
        self.wo = nn.Linear(config.ff_mult * config.dim, config.dim)
        nn.init.zeros_(self.wo.weight)

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.wo(F.silu(self.w1(x)) * self.w2(x))


class SelfAttention(nn.Module):
    """Basic multiheaded attention block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.dim % config.num_heads == 0
        self.num_heads = config.num_heads
        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.out = nn.Linear(config.dim, config.dim, bias=False)
        self.scale = nn.Parameter(T.tensor(1.0))
        nn.init.zeros_(self.out.weight)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Dispatch to the appropriate attention function based on the inputs."""
        B, S, D = x.shape
        HD = D // self.num_heads
        NH = self.num_heads
        q, k, v = self.qkv(x).view(B, S, 3, NH, HD).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = norm(q), norm(k) * self.scale
        a_out = F.scaled_dot_product_attention(q, k, v)
        a_out = a_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out(a_out)


class Block(nn.Module):
    """Building block for the Transformer Encoder containing MHSA and FFN."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.sa = SelfAttention(config)
        self.ff = SwiGLUNet(config)

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = x + self.sa(norm(x))
        x = x + self.ff(norm(x))
        return x  # noqa: RET504


class TransformerClassifier(nn.Module):
    """Simple transformer stack for class classification."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.class_tokens > 0
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.lin_out = nn.Linear(config.dim, config.output_dim)
        self.class_tokens = nn.Parameter(T.randn(1, config.class_tokens, config.dim))
        self.pos_enc = nn.Parameter(T.randn(config.seq_len, config.dim))

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Pass through all layers of the transformer."""
        B, _S = x.shape
        x = self.embed(x) + self.pos_enc
        x = T.cat((self.class_tokens.expand(B, -1, -1), x), dim=1)
        x = norm(x)
        for layer in self.layers:
            x = layer(x)
        x = x[:, 0]  # Trim off the class token (others are registers)
        return self.lin_out(norm(x))
