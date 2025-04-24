import torch
import torch.nn as nn
import numpy as np


from mlps import MLP1A, MLP1B
    
class SpectralEncoder(nn.Module):
    """
    First part of the presented architecture.
    Yields to a spatio-spectral embedding at time t
    """

    def __init__(self, device):
        super(SpectralEncoder, self).__init__()
        self.device = device
        
        self.mlp1a = MLP1A()
        self.mlp1b = MLP1B()
        

    def forward(self, x):  # x: [batch_size x  seq_len x channels:10]

        
        batch_size, seq_len, channels = x.shape
        

        mlp1_output = self.mlp1a(x)                              # [batch_size x seq_len x hidden_state:32]
        mlp1_output = self.mlp1b(mlp1_output)                   # [batch_size x seq_len x hidden_state:64]


        pooled = mlp1_output.contiguous().view(batch_size, seq_len, -1)  # [batch_size x seq_len x hidden_state:64]
        pooled = pooled.type('torch.FloatTensor')

        return pooled
    
    def forward_lrp(self, x):
        batch_size, seq_len, channels = x.shape
        

        mlp1_output = self.mlp1a.forward_lrp(x)                              # [batch_size x seq_len x hidden_state:32]
        mlp1_output = self.mlp1b(mlp1_output)                   # [batch_size x seq_len x hidden_state:64]


        pooled = mlp1_output.contiguous().view(batch_size, seq_len, -1)  # [batch_size x seq_len x hidden_state:64]
        pooled = pooled.type('torch.FloatTensor')

        return pooled