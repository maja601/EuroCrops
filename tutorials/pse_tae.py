
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pse import SpectralEncoder
from attention_layer import AttentionLayer
from tae import TAE

from positional_encoding import PositionalEncoding

class PSE_TAE(nn.Module):

    def __init__(self, device, heads=4, d_e= 64):
        super(PSE_TAE, self).__init__()
        self.spectral_encoder = SpectralEncoder(device)
        
  
            
        self.d_e = d_e
        self.d_k = d_e // heads
        self.h = heads

        self.attn = AttentionLayer(self.d_k,self.h)
        self.tae = TAE()


             
        self.R0 = ()

    def forward(self, x, days):
        encoding = self.spectral_encoder(x)
        

        batch_size, seq_len, hidden_state = encoding.size()

        pos_encoding = PositionalEncoding(days[0,:,:].squeeze(0), d_e = self.d_e, max_len=seq_len)

        e_p = pos_encoding(encoding)
        # Queries
        attention_output1 = self.attn(e_p, batch_size,seq_len)
        attention_output = attention_output1.contiguous().view(batch_size, -1) # batch_size x seq_len * dk * num_heads (d_e 64)
        # Output
        batch_size, seq_len = attention_output.size()
        output = self.tae(attention_output)

        return output
    

    # adapted from [Ali et al., 2022](https://arxiv.org/abs/2202.07304)
    def forward_and_explain(self, x, label_id, days):
        
        
        batch_size, seq_len, channels = x.shape
        x.requires_grad_(True) 
        x.retain_grad()
        


        encoding = self.spectral_encoder.forward_lrp(x)

        batch_size, seq_len, hidden_state = encoding.size()
        
        pos_encoding = PositionalEncoding(days[0,:,:].squeeze(0),device = self.device, d_e = self.d_e, max_len=seq_len)
        
        e_p = pos_encoding(encoding).to(self.device)        #[batch_size x seq_len x hidden_state:64]
        # Queries
        attention_output1 = self.attn1.forward_lrp(e_p, batch_size,seq_len)

        attention_output = attention_output1.contiguous().view(batch_size, -1) # batch_size x seq_len * dk * num_heads (d_e 64)

        # Output
        batch_size, seq_len = attention_output.size()
        output = self.tae.forward_lrp(attention_output)

        Rout = output[:,label_id]
        
        self.R0 = Rout.detach().cpu().numpy()
        
        Rout.backward()


        R_grad = x.grad
        R_attn =  (R_grad)*x
        R_ = R_attn.cpu()

        R = R_.detach().numpy()
        
        return {'logits': output, 'R': R}