
import math
import torch
import torch.nn as nn
 

class PositionalEncoding(nn.Module):

    def __init__(self,days, d_e=64, max_len=80):
        super(PositionalEncoding, self).__init__()        

        # Calculate the positional encoding p
        p = torch.zeros(max_len, d_e)
        
        div_term = torch.exp(torch.arange(0, d_e, 2).float() * (-math.log(1000.0) / d_e))
        p[:, 0::2] = torch.sin(days * div_term)
        p[:, 1::2] = torch.cos(days * div_term)
        p = p.unsqueeze(0)
        
        self.register_buffer('p', p)

    def forward(self, x):
        
        x = x + self.p
        return x