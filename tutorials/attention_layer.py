import torch
import torch.nn as nn
import numpy as np
import math
from lrp_ln import LRPLayerNorm
import torch.nn.functional as F
"""
Attention layer based on Garnot and Marc's implementation
forward_lrp drops the dropout layer + detaches softmax and variance(in the layer norm) as per LRP propagation rules
"""



class AttentionLayer(nn.Module):
    def __init__(self, d_k, h):
        super(AttentionLayer, self).__init__()
        self.h = h
        self.d_k = d_k
        self.d_e = self.d_k * self.h

        self.fc1_q = nn.Linear(self.d_e, self.d_e)
        self.fc1_k = nn.Linear(self.d_e, self.d_e)
        self.fc1_v = nn.Linear(self.d_e, self.d_e)

        self.fc1 = nn.Linear(self.d_k*self.h, self.d_e)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(self.d_k*self.h)


        self.q = ()
        self.k = ()
        self.v = ()
        self.attention_scores = ()
        self.attention_output = ()
        self.attention_probs = ()


    def forward(self, e_p, batch_size, seq_len):
        q = self.fc1_q(e_p)         # [batch_size x seq_len x hidden_state:64]
        q = q.view(batch_size,  seq_len, self.h, self.d_k)     # [batch_size x seq_len x num_heads x d_k]
        q = q.permute(0, 2, 1, 3).contiguous().view(-1,seq_len,  self.d_k)     # [batch_size * num_heads x seq_len x d_k]
        self.q = q
        # Keys
        k = self.fc1_k(e_p)                                                 # [batch_size x seq_len x hidden_state:64]
        k = k.view(batch_size, seq_len, self.h, self.d_k)                   # [batch_size x seq_len x num_heads x d_k]
        k = k.permute(0, 2, 1, 3).contiguous().view(-1, seq_len, self.d_k)  # [batch_size * num_heads x seq_len x d_k]
        self.k = k
        # Values 
          
        v = self.fc1_v(e_p)  # [batch_size * num_heads x seq_len x hidden:64]
        v = v.view(batch_size, seq_len, self.h, self.d_k)       # [batch_size x seq_len x num_heads x d_k]
        v = v.permute(0, 2, 1, 3).contiguous().view(-1, seq_len, self.d_k) # [batch_size * num_heads x seq_len x d_k]
        self.v = v
        
        # Attention
        attention_scores = q.matmul(k.transpose(-2, -1)) / math.sqrt(self.d_k)      # [batch_size * num_heads x seq_len x seq_len]
        self.attention_scores = attention_scores        # ,4,80,80  batch_size * num_heads x seq_len x seq_len
        
        attention_probs = F.softmax(attention_scores, dim=-1)               # [batch_size * num_heads x 64 (d_e) x seq_len]
       
        attention_output = torch.matmul(attention_probs, v).squeeze()       #  [batch_size* 4 (num_h) x 80 (seq length) x 64 (d_e)]
        
        
        attention_output = attention_output.view(self.h, batch_size,seq_len, self.d_k)  # num_heads x batch_size x seq_len x dk 16
        attention_output = attention_output.permute(1,2,0,3).contiguous().view(batch_size,seq_len,-1) # batch_size x seq_len x dk * num_heads (d_e 64)


        ################ adding drop out and layer norm to reduce overfitting
        attention_output = self.dropout(self.fc1(attention_output)) # batch_size x seq_len x dk * num_heads (d_e 64)
        attention_output = self.layer_norm(attention_output+e_p) # batch_size x seq_len x dk * num_heads (d_e 64)
        
        self.attention_output = attention_output
        self.attention_probs = attention_probs

        return attention_output 

    # For LRP, special layernorm + no dropout + detaching     
    def forward_lrp(self, e_p, batch_size, seq_len):
        q = self.fc1_q(e_p)         # [batch_size x seq_len x hidden_state:64]
        q_hat = q.view(batch_size,  seq_len, self.h, self.d_k)     # [batch_size x seq_len x num_heads x d_k]
        q_hat = q_hat.permute(0, 2, 1, 3).contiguous().view(-1,seq_len,  self.d_k)     # [batch_size * num_heads x seq_len x d_k]

        
        # Keys
        k = self.fc1_k(e_p)                                                 # [batch_size x seq_len x hidden_state:64]
        k = k.view(batch_size, seq_len, self.h, self.d_k)                   # [batch_size x seq_len x num_heads x d_k]
        k = k.permute(0, 2, 1, 3).contiguous().view(-1, seq_len, self.d_k)  # [batch_size * num_heads x seq_len x d_k]
      
        
        # Values  
        v = self.fc1_v(e_p)  # [batch_size * num_heads x seq_len x hidden:64]
        v = v.view(batch_size, seq_len, self.h, self.d_k)       # [batch_size x seq_len x num_heads x d_k]
        v = v.permute(0, 2, 1, 3).contiguous().view(-1, seq_len, self.d_k) # [batch_size * num_heads x seq_len x d_k]
 


        # Attention
        attention_scores = q_hat.matmul(k.transpose(-2, -1)) / math.sqrt(self.d_k)      # [batch_size * num_heads x 128 x seq_len]
        self.attention_scores = attention_scores        # ,4,80,80  batch_size * num_heads x seq_len x seq_len
        
        attention_probs = F.softmax(attention_scores, dim=-1).detach()               # [batch_size * num_heads x 128 x seq_len]
       
        attention_output = torch.matmul(attention_probs, v).squeeze()       # [batch_size * num_heads x hidden_state:128] [batch_size*4*80*64]

        attention_output = attention_output.view(self.h, batch_size,seq_len, self.d_k)  # num_heads x batch_size x seq_len x dk 16
        attention_output = attention_output.permute(1,2,0,3).contiguous().view(batch_size,seq_len,-1) # batch_size x seq_len x dk * num_heads (d_e 64)
        
        layer_normLRP = LRPLayerNorm(self.d_e)
        
        
        attention_output = self.fc1(attention_output)

        attention_output = layer_normLRP(attention_output+e_p) # batch_size x seq_len x dk * num_heads (d_e 64)
        attention_output = self.layer_norm.weight * attention_output + self.layer_norm.bias


        return attention_output