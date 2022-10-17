import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import default_dtype_torch
from base import BaseModel
from numpy import arange

#from utils import gen_all_binary_vectors
def gen_all_binary_vectors(length):
    return ((torch.arange(2**length).unsqueeze(1) >> torch.arange(length - 1, -1, -1)) & 1).float()

class PositionalEncoding(nn.Module):
    def __init__(self, n, d_model):
        super().__init__()
        den = torch.exp(- torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pos = torch.arange(0, n).reshape(n, 1)
        pos_embedding = torch.zeros((n, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        return x + self.pos_embedding


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, n, d_model):
        super().__init__()
        self.positional_embedding = nn.Embedding(n, d_model)
        positions = torch.arange(n)
        self.register_buffer('positions', positions)

    def forward(self, x):
        return x + self.positional_embedding(self.positions)


class TraDE(BaseModel):
    """
    Transformers for density estimation or stat-mech problems
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        #self.n = kwargs['n']
        self.L = kwargs['L']
        self.M = kwargs['M']
        self.bits = kwargs['bits']
        self.n = self.L
        self.d_model = kwargs['d_model']
        self.d_ff = kwargs['d_ff']
        self.n_layers = kwargs['n_layers']
        self.n_heads = kwargs['n_heads']
        self.device = kwargs['device']
        self.MConstrain = kwargs['MConstrain']
        
        #self.fc_in = nn.Embedding(2, self.d_model)
        self.fc_in = nn.Embedding(self.M, self.d_model)
        # self.positional_encoding = PositionalEncoding(self.n, self.d_model)
        self.positional_encoding = LearnablePositionalEncoding(self.n, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=self.d_ff,
                                                   dropout=0,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.n_layers)
        #self.fc_out = nn.Linear(self.d_model, 1)
        self.fc_out = nn.Linear(self.d_model, self.M)

        self.register_buffer('mask', torch.ones(self.n, self.n))
        self.mask = torch.tril(self.mask)
        self.mask = self.mask.masked_fill(self.mask == 0, float('-inf'))

    def forward(self, x):
        # print(x)
        x = torch.cat((torch.ones(x.shape[0], 1, device=self.device), x[:, :-1]), dim=1)
        x = F.relu(self.fc_in(x.int()))  # (batch_size, n, d_model)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask=self.mask)
        x=  self.fc_out(x)
        if self.MConstrain[0]==0: #If no number constrain
            x_hat = F.log_softmax(x, dim=2) # F.softmax(x, dim=2)
        else:                      #Number constrain
            x_hat=torch.ones_like(x)*(-100)
            index1=arange(self.n)[self.MConstrain<self.M]
            index0=arange(self.n)[self.MConstrain==self.M]
            index11=self.MConstrain[index1[0]]
            index00=self.MConstrain[index0[0]]
            x_hat[:,index0,:index00] = F.log_softmax(x[:,index0,:index00], dim=2) 
            x_hat[:,index1,:index11] = F.log_softmax(x[:,index1,:index11], dim=2)
            
        
        
        return x_hat

    def log_prob(self, x):
        x_hat = self(x)
        log_prob = torch.zeros_like(x)#[Batchsize,1,SpinNum]
        for i in range(self.L):
            ids = x[:,i].long()
            log_prob[:,i] = x_hat[:,i,:].gather(1, ids.view(-1,1))[:,0]
            
        log_prob = log_prob.sum(dim=1) # It seems to sum up all the log(conditional P)
        #Check the probability data dimension: [Batchsize]
        return log_prob
    
        

    def sample(self, batch_size):
        samples = torch.randint(0, self.M, size=(batch_size, self.n), dtype=default_dtype_torch, device=self.device)
        for i in range(self.n):
            x_hat = self(samples)
            if self.MConstrain[0]==0:
                samples[:, i] = torch.multinomial(
                        torch.exp(x_hat[:, i,:]), 1).to(default_dtype_torch)[:,0]#torch.bernoulli(x_hat[:, i])                                
            else:
                samples[:, i] = torch.multinomial(
                        torch.exp(x_hat[:, i,:self.MConstrain[i]]), 1).to(default_dtype_torch)[:,0]#torch.bernoulli(x_hat[:, i])    
        return samples, x_hat


if __name__ == '__main__':
    kwargs_dict = {
        'n': 4,
        'd_model': 64,
        'd_ff': 128,
        'n_layers': 2,
        'n_heads': 2,
        'device': 'cpu'
    }

    model = TraDE(**kwargs_dict).to(kwargs_dict['device'])
    print(model)

    # test normalization condition
    x = gen_all_binary_vectors(kwargs_dict['n']).to(kwargs_dict['device'])
    log_prob = model.log_prob(x)
    print(log_prob.exp().sum())

    # test sampling
    y = model.sample(10)
    print(y)
