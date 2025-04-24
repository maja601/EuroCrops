import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlps import MLP2A,MLP2B,MLP3A,MLP3B

class TAE(nn.Module):
    def __init__(self) -> None:
        super(TAE, self).__init__()
        self.mlp2a = MLP2A()
        self.mlp2b = MLP2B()
        self.mlp3a = MLP3A()
        self.decoder = MLP3B()


    def forward(self, attention_output):    
        
        o_hat = self.mlp2a(attention_output)
        o_hat = self.mlp2b(o_hat)
        o_hat = self.mlp3a(o_hat)

        
        output = self.decoder(o_hat)
        return output
    
    def forward_lrp(self, attention_output):    
        
        o_hat = self.mlp2a.forward_lrp(attention_output)
        o_hat = self.mlp2b(o_hat)
        o_hat = self.mlp3a(o_hat)

        output = self.decoder(o_hat)
        return output