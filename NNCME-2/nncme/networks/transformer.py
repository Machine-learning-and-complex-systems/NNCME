import math
from nncme.args import args
import torch
import torch.nn as nn
import torch.nn.functional as F
from nncme.networks.base import BaseModel
from nncme.utils import default_dtype_torch, scaled_dot_product_attention
from numpy import arange

def gen_all_binary_vectors(length):
    """Gen all binary vectors operation.
    Args:
        length: Parameter forwarded to gen_all_binary_vectors.
    Returns:
        Result produced by gen_all_binary_vectors.
    """


    return ((torch.arange(2**length).unsqueeze(1) >> torch.arange(length - 1, -1, -1)) & 1).float()

class PositionalEncoding(nn.Module):
    """Representation of the PositionalEncoding component.
    """


    def __init__(self, n, d_model):
        """  init   operation.
        Args:
            n: Parameter forwarded to __init__.
            d_model: Parameter forwarded to __init__.
        """


        super().__init__()
        den = torch.exp(- torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pos = torch.arange(0, n).reshape(n, 1)
        pos_embedding = torch.zeros((n, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        """Forward operation.
        Args:
            x: Parameter forwarded to forward.
        Returns:
            Result produced by forward.
        """
        return x + self.pos_embedding


class LearnablePositionalEncoding(nn.Module):
    """Representation of the LearnablePositionalEncoding component.
    """


    def __init__(self, n, d_model):
        """  init   operation.
        Args:
            n: Parameter forwarded to __init__.
            d_model: Parameter forwarded to __init__.
        """


        super().__init__()
        self.positional_embedding = nn.Embedding(n, d_model)
        positions = torch.arange(n)
        self.register_buffer('positions', positions)

    def forward(self, x):
        """Forward operation.
        Args:
            x: Parameter forwarded to forward.
        Returns:
            Result produced by forward.
        """


        return x + self.positional_embedding(self.positions)


class CustomMultiheadAttention(nn.Module):
    """Representation of the CustomMultiheadAttention component.
    """


    def __init__(self, d_model, n_heads):
        """  init   operation.
        Args:
            d_model: Parameter forwarded to __init__.
            n_heads: Parameter forwarded to __init__.
        """


        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.last_attn = None  

    def forward(self, x, mask):
        """Forward operation.
        Args:
            x: Parameter forwarded to forward.
            mask: Parameter forwarded to forward.
        Returns:
            Result produced by forward.
        """


        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)


        scale_factor = 1 / math.sqrt(self.d_k)
        attn_score = (q @ k.transpose(-2, -1)) * scale_factor  # (B, h, T, T)
        if mask is not None:
            attn_score = attn_score + mask.unsqueeze(0).unsqueeze(0)  # broadcast mask
        attn_weight = torch.softmax(attn_score, dim=-1)
        self.last_attn = attn_weight.detach()

        y = attn_weight @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    """Representation of the TransformerBlock component.
    """


    def __init__(self, d_model, n_heads, d_ff):
        """  init   operation.
        Args:
            d_model: Parameter forwarded to __init__.
            n_heads: Parameter forwarded to __init__.
            d_ff: Parameter forwarded to __init__.
        """


        super().__init__()
        self.attn = CustomMultiheadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        """Forward operation.
        Args:
            x: Parameter forwarded to forward.
            mask: Parameter forwarded to forward.
        Returns:
            Result produced by forward.
        """


        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class TraDE(BaseModel):
    """Representation of the TraDE component.
    """


    def __init__(self, *args, **kwargs):
        """  init   operation.
        Args:
            *args: Parameter forwarded to __init__.
            **kwargs: Parameter forwarded to __init__.
        """


        super().__init__()
        self.L = kwargs['L']
        self.M = kwargs['M']
        # self.bits = kwargs['bits']
        self.n = self.L
        self.d_model = kwargs['d_model']
        self.d_ff = kwargs['d_ff']
        self.n_layers = kwargs['n_layers']
        self.n_heads = kwargs['n_heads']
        self.device = kwargs['device']
        self.MConstrain = kwargs['MConstrain']
        if self.MConstrain[0]==0: self.constrain=False
        else: self.constrain=True

        self.fc_in = nn.Embedding(self.M, self.d_model)
        self.positional_encoding = LearnablePositionalEncoding(self.n, self.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(self.d_model, self.n_heads, self.d_ff) for _ in range(self.n_layers)])
        self.fc_out = nn.Linear(self.d_model, self.M)

        self.register_buffer('mask', torch.ones(self.n, self.n))
        self.mask = torch.tril(self.mask)
        self.mask = self.mask.masked_fill(self.mask == 0, float('-inf'))

    def _forward(self, x):
        """ forward operation.
        Args:
            x: Samples
        """
        x = torch.cat((torch.ones(x.shape[0], 1, device=self.device), x[:, :-1]), dim=1)
        x = F.relu(self.fc_in(x.int()))
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x, mask=self.mask)
        x = self.fc_out(x)

        if self.MConstrain[0] == 0:
            x_hat = F.log_softmax(x, dim=2)
        else:
            x_hat = torch.ones_like(x) * -100
            index1 = arange(self.n)[self.MConstrain < self.M]
            index0 = arange(self.n)[self.MConstrain == self.M]
            index11 = self.MConstrain[index1[0]]
            index00 = self.MConstrain[index0[0]]
            x_hat[:, index0, :index00] = F.log_softmax(x[:, index0, :index00], dim=2)
            x_hat[:, index1, :index11] = F.log_softmax(x[:, index1, :index11], dim=2)

        return x_hat

    def forward(self, x):
        """Forward operation.
        Args:

            x: Parameter forwarded to forward.
        Returns:
            Result produced by forward.
        """


        x_hat = self._forward(x)  # (B, L, M)
        ids = x.long().unsqueeze(-1)  # (B, L, 1)
        log_prob = torch.gather(x_hat, 2, ids).squeeze(-1)  # (B, L)
        return log_prob.sum(dim=1)

    def log_prob(self, x):
        """Log prob operation.
        Args:
            x: samples.
        Returns:
            log_prob of x.
        """
        return self.forward(x)
    

    def log_prob_diff(self, x, kernel_size=None):
        """Log prob diffusive operation.
        Args:
            x: samples.
            kernel_size: uniform convolution kernel size.
        Returns:
            Result produced by log_prob_diff.

        """
        x_hat = self._forward(x) # (B, L, M)
        kernel_size=args.kernel
        log_probs = []
        for i in range(self.n):
            logits = x_hat[:, i, :] # (B, M)
            if self.constrain:
                logits = logits[:, :self.MConstrain[i]]
            probs = F.softmax(logits, dim=1)
            # Diffusive kernel
            kernel = torch.ones(1, 1, kernel_size, device=probs.device) / kernel_size
            # Pad to maintain same length
            pad_left = (kernel_size - 1) // 2
            pad_right = kernel_size - 1 - pad_left
            probs_padded = F.pad(probs.unsqueeze(1), (pad_left, pad_right), mode='reflect')
            probs_d = F.conv1d(probs_padded, kernel).squeeze(1)

            probs_d = torch.clamp(probs_d, min=1e-12)
            probs_d = probs_d / probs_d.sum(dim=1, keepdim=True)
    
            ids = x[:, i].long()
            log_p = torch.log(probs_d + 1e-12).gather(1, ids.view(-1, 1))[:, 0]
            log_probs.append(log_p)
        return torch.stack(log_probs, dim=1).sum(dim=1)

    def _forward_alpha(self, x):
        """ forward alpha operation.
        Args:
            x: samples.
        Returns:
            Result produced by _forward_alpha.
        """

        x_hat = self._forward(x)  # (B, L, M)
        alpha = args.alpha
        log_probs = []
        for i in range(self.n):
            logits = x_hat[:, i, :] # (B, M)
            if self.constrain:
                logits = logits[:, :self.MConstrain[i]]
            logq_i = F.log_softmax(alpha * logits, dim=1)
            ids = x[:, i].long()
            log_probs.append(logq_i.gather(1, ids.view(-1, 1))[:, 0])
        return torch.stack(log_probs, dim=1).sum(dim=1)

    def log_prob_alpha(self, x):
        """Log prob alpha operation.
        Args:
            x: samples.
        Returns:
            Result produced by log_prob_alpha.
        """
        return self._forward_alpha(x)

    @torch.no_grad()
    def sample(self, batch_size):
        """Sample operation.
        Args:
            batch_size: samples number.
        Returns:
            Samples.
        """
        samples = torch.randint(0, self.M, size=(batch_size, self.n), dtype=default_dtype_torch, device=self.device)
        for i in range(self.n):
            x_hat = self._forward(samples)
            logits = x_hat[:, i, :]
            if self.constrain:
                logits = logits[:, :self.MConstrain[i]]
            probs = F.softmax(logits, dim=1)
            samples[:, i] = torch.multinomial(probs, 1).to(default_dtype_torch)[:, 0]
        return samples, torch.zeros_like(samples)
    

    
    def sampleDiffusive(self, batch_size, kernel_size=None):
        """Sample diffusive operation.
        """
        kernel_size=args.kernel
        x = torch.zeros(batch_size, self.n, dtype=default_dtype_torch, device=self.device)
        for i in range(self.n):
            x_hat = self._forward(x)
            logits = x_hat[:, i, :]
            if self.constrain:
                logits = logits[:, :self.MConstrain[i]]
            probs = F.softmax(logits, dim=1)
    
            # Diffusive kernel
            kernel = torch.ones(1, 1, kernel_size, device=probs.device) / kernel_size
            
            # Pad to maintain same length
            pad_left = (kernel_size - 1) // 2
            pad_right = kernel_size - 1 - pad_left
            probs_padded = F.pad(probs.unsqueeze(1), (pad_left, pad_right), mode='reflect')
            probs_d = F.conv1d(probs_padded, kernel).squeeze(1)

            # Normalize and ensure numerical stability
            probs_d = torch.clamp(probs_d, min=1e-12)
            probs_d = probs_d / probs_d.sum(dim=1, keepdim=True)
    
            x[:, i] = torch.multinomial(probs_d, 1).squeeze(-1)
        return x, torch.zeros_like(x)
    
    @torch.no_grad()
    def sampleAlpha(self, batch_size): 
        """Samplealpha operation.
        """
        samples = torch.randint(0, self.M, size=(batch_size, self.n), dtype=default_dtype_torch, device=self.device)
        alpha = args.alpha
        for i in range(self.n):
            x_hat = self._forward(samples)
            logits = x_hat[:, i, :]
            if self.constrain:
                logits = logits[:, :self.MConstrain[i]]
            probs_powered = F.softmax(alpha*logits,dim=1)
            samples[:, i] = torch.multinomial(probs_powered, 1).to(default_dtype_torch)[:, 0]
        return samples, torch.zeros_like(samples)
