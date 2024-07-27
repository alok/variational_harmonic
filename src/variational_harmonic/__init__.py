# %%
from pathlib import Path
import torch
from torch import Tensor
from jaxtyping import Float

from typing import NamedTuple
from torch import nn
from torch.nn import Linear, Sequential, GELU


class EncOut(NamedTuple):
    mean: Tensor
    std: Tensor


class Encoder(nn.Module):
    def __init__(self):
        self.mean_head = Sequential(
            Linear(10, 10),
            GELU(),
            Linear(10, 10),
        )
        self.covar_head = Sequential(
            Linear(10, 10),
            GELU(),
            Linear(10, 10),
        )

    def forward(self, x):
        mean = self.mean_head(x)
        co_var = self.covar_head(x)
        std = torch.exp(co_var / 2)
        return EncOut(mean=mean, std=std)



class VAE(nn.Module):
    def __init__(self, in_shape: int, out_shape: int) -> None:
        self.encoder = Encoder()
        self.mv_normal_params = []
        self.decoder = Sequential(Linear(10, 10))

    def forward(self, x: Tensor) -> Tensor:
        mean, std = self.encoder(x)
        z = self.reparam(mean, std)

        decoded = self.decoder(z)
        return decoded

    def reparam(self, mean, std):
        eps = torch.randn_like(mean)
        z = mean + std * eps
        return z



# enc
# dec
# reparam
# loss


def hello(_):
    a = 2
    return "Hello from variational-harmonic!"


a = hello(aa := 2 + 2)
# %%
