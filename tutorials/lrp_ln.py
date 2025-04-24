
# special LayerNorm layer for LRP 
# from [Ali et al., 2022](https://arxiv.org/abs/2202.07304)


import torch
from torch import nn, Size

class LRPLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-5):
                 
                 super().__init__()
                 
                 if isinstance(normalized_shape, int):
                    normalized_shape = torch.Size([normalized_shape])
                 elif isinstance(normalized_shape, list):
                    normalized_shape = torch.Size(normalized_shape)
                
                 
                 self.normalized_shape = normalized_shape
                 self.eps = eps


    def forward(self, x: torch.Tensor):
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]


        mean = x.mean(-1, keepdim=True)

        var = x.var(-1, keepdim = True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps).detach()


        return x_norm    
