import torch
import torch.nn.functional as F
from torch import nn
from nncme.args import args

from nncme.networks.base import BaseModel
from nncme.utils import default_dtype_torch


class NADE(BaseModel):
    """NADE network
    """


    def __init__(self, z2=False, *args, **kwargs):
        """  init   operation.
        Args:
            *args: Parameter forwarded to __init__.
            **kwargs: Parameter forwarded to __init__.
        """


        super().__init__()
        self.L = kwargs['L']
        self.M = kwargs['M']  # M is the number of possible states per element
        self.net_width = kwargs['net_width']
        self.MConstrain= kwargs['MConstrain']
        self.epsilon = torch.tensor(kwargs['epsilon'], 
                                dtype=default_dtype_torch,
                                device=kwargs['device']) 
        if self.MConstrain[0]==0: self.constrain=False
        else: self.constrain=True
        self.conservation = torch.as_tensor(kwargs['conservation'],device=kwargs['device'])
        n = self.L
        hidden_dim = self.net_width
        self.n = n
        self.hidden_dim = hidden_dim
        self.device = kwargs['device']
        self.dtype = default_dtype_torch#torch.float64 if dtype == "float64" else torch.float32
        self.z2 = z2
        print('NADE self.dtype:',self.dtype)
        
        self.register_parameter("W", nn.Parameter(torch.randn(hidden_dim, n, dtype=self.dtype)))
        self.register_parameter("c", nn.Parameter(torch.zeros(hidden_dim, dtype=self.dtype)))
        self.register_parameter("V", nn.Parameter(torch.randn(n * self.M, hidden_dim, dtype=self.dtype)))
        self.register_parameter("b", nn.Parameter(torch.zeros(n * self.M, dtype=self.dtype)))
        
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.V)


        
    def _forward(self, x):
        """ forward operation.
        Args:
            x: Samples.
        Returns:
            log_prob of each sample.
        """
        x = x.to(dtype=self.dtype, device=self.device)
        log_prob_list = []
        for i in range(self.n):
            h_i = torch.sigmoid(self.c + torch.einsum("hi,bi->bh", self.W[:, :i], x[:, :i]))
            logits = self.b[i * self.M:(i + 1) * self.M] + torch.einsum("oh,bh->bo", self.V[i * self.M:(i + 1) * self.M], h_i)#.view(-1, self.M)
            if self.constrain:
                log_prob1=F.log_softmax(logits[:,:self.MConstrain[i]],dim=1)
            else:
                log_prob1= F.log_softmax(logits,dim=1)#(500,20)
            ids=x[:,i].long()
            log_prob=log_prob1.gather(1, ids.view(-1,1))[:,0] #torch.gather(log_prob1, 1, x1[:,i].unsqueeze(1))#.squeeze(2)
            log_prob_list.append(log_prob)
        return torch.stack(log_prob_list, dim=1)#torch.stack(logits_list, dim=1)

    def forward(self, x):
        """Forward operation.
        Args:
            x: Samples.
        Returns:
            log_prob of each sample.
        """
        log_prob=self._forward(x)
        return log_prob.sum(-1)


    def log_prob(self, x):
        """Log prob operation.
        Args:
            x: Parameter forwarded to log_prob.
        Returns:
            Result produced by log_prob.
        """
        log_prob=self._forward(x)
        return log_prob.sum(-1)
    


    @torch.no_grad()
    def _sample(self, batch_size):
        """ sample operation.
        Args:
            batch_size: Parameter forwarded to _sample.
        Returns:
            Result produced by _sample.
        """
        x = torch.zeros(batch_size, self.n, dtype=self.dtype, device=self.device)
        for i in range(self.n):
            h_i = torch.sigmoid(self.c + torch.einsum("hi,bi->bh", self.W[:, :i], x[:, :i]))
            logits = self.b[i * self.M:(i + 1) * self.M] + torch.einsum("oh,bh->bo", self.V[i * self.M:(i + 1) * self.M], h_i)#.view(-1, self.M)
            if self.constrain:
                probs=F.softmax(logits[:,:self.MConstrain[i]],dim=1)
            else:
                probs= F.softmax(logits,dim=1) #(bacth_size,inputDim)
            x[:, i] = torch.multinomial(probs, 1).squeeze(-1)
        x_hat = torch.zeros_like(x) #my code
        return x, x_hat
    
    @torch.no_grad()
    def sample(self, batch_size):
        """Sample from original distribution.
        """
        return self._sample(batch_size)


    @torch.no_grad()
    def sampleDiffusive(self, batch_size, kernel_size=None):
        """Sample from diffusive distribution.
        """
        kernel_size=args.kernel
        x = torch.zeros(batch_size, self.n, dtype=self.dtype, device=self.device)
        for i in range(self.n):
            h_i = torch.sigmoid(self.c + torch.einsum("hi,bi->bh", self.W[:, :i], x[:, :i]))
            logits = self.b[i * self.M:(i + 1) * self.M] + torch.einsum(
                "oh,bh->bo", self.V[i * self.M:(i + 1) * self.M], h_i)
    
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

    def log_prob_diff(self, x, kernel_size=None):
        """Log prob diffusive operation.
        """
        kernel_size=args.kernel
        log_probs = []
        for i in range(self.n):
            h_i = torch.sigmoid(self.c + torch.einsum("hi,bi->bh", self.W[:, :i], x[:, :i]))
            logits = self.b[i * self.M:(i + 1) * self.M] + torch.einsum(
                "oh,bh->bo", self.V[i * self.M:(i + 1) * self.M], h_i)
    
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


    def sampleAlpha(self, batch_size):
        """Sample from alpha distribution.
        """
        alpha = args.alpha#self.alpha_value()   
        x = torch.zeros(batch_size, self.n, dtype=self.dtype, device=self.device)
        for i in range(self.n):
            h_i = torch.sigmoid(self.c + torch.einsum("hi,bi->bh", self.W[:, :i], x[:, :i]))
            logits = self.b[i*self.M:(i+1)*self.M] + torch.einsum("oh,bh->bo", self.V[i*self.M:(i+1)*self.M], h_i)
            if self.constrain:
                logits = logits[:, :self.MConstrain[i]]
            probs_alpha = F.softmax(alpha * logits, dim=1)           
            x[:, i] = torch.multinomial(probs_alpha, 1).squeeze(-1)
        return x, torch.zeros_like(x)
    
    
    def log_prob_alpha(self, x):
        """Log prob alpha operation.
        """
        alpha = args.alpha#self.alpha_value()   
        log_probs = []
        for i in range(self.n):
            h_i = torch.sigmoid(self.c + torch.einsum("hi,bi->bh", self.W[:, :i], x[:, :i]))
            logits = self.b[i*self.M:(i+1)*self.M] + torch.einsum("oh,bh->bo", self.V[i*self.M:(i+1)*self.M], h_i)
            if self.constrain:
                logits = logits[:, :self.MConstrain[i]]
            logq_i = F.log_softmax(alpha * logits, dim=1)           
            ids = x[:, i].long()
            log_probs.append(logq_i.gather(1, ids.view(-1,1))[:,0])
        return torch.stack(log_probs, dim=1).sum(dim=1)
