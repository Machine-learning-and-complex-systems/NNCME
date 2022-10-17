# MADE: Masked Autoencoder for Distribution Estimation

import torch
from numpy import log
from numpy import arange
from torch import nn

from pixelcnn import ResBlock
from utils import default_dtype_torch
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, n, bias, exclusive):
        super(MaskedLinear, self).__init__(in_channels * n, out_channels * n,
                                           bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
        self.exclusive = exclusive

        #what is register_buffer? They are parameters without doing differentiation
        #should revise size? here it is masking the connection matrix between input and output, which should n X n
        #but below we may need revise it size to 1D
        self.register_buffer('mask', torch.ones([self.n] * 2)) #My code
        #print(torch.ones([self.n] ))
        

        if self.exclusive:
            self.mask = 1 - torch.triu(self.mask)
            #lower triangle without diagonal 
        else:
            self.mask = torch.tril(self.mask)
            #lower triangle with diagonal 
            
        self.mask=self.mask-torch.tril(self.mask,diagonal=-4) # My code: Only use neighbors, manual.
        
        #self.mask=torch.ones([self.n] * 2) #My code
        
        self.mask = torch.cat([self.mask] * in_channels, dim=1)
        #print(self.mask )
        self.mask = torch.cat([self.mask] * out_channels, dim=0)
        self.weight.data *= self.mask

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)

    def extra_repr(self):
        return (super(MaskedLinear, self).extra_repr() +
                ', exclusive={exclusive}'.format(**self.__dict__))


# TODO: reduce unused weights, maybe when torch.sparse is stable
class ChannelLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, n, bias):
        super(ChannelLinear, self).__init__(in_channels * n, out_channels * n,
                                            bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n

        self.register_buffer('mask', torch.eye(self.n))
        self.mask = torch.cat([self.mask] * in_channels, dim=1)
        self.mask = torch.cat([self.mask] * out_channels, dim=0)
        self.weight.data *= self.mask

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)


class MADE1D(nn.Module):
    def __init__(self, **kwargs):
        super(MADE1D, self).__init__()
        self.L = kwargs['L']
        self.n = self.L#**2  # My code: Number of sites
        self.net_depth = kwargs['net_depth']
        self.net_width = kwargs['net_width']
        self.bias = kwargs['bias']
        self.z2 = kwargs['z2']
        self.binary = kwargs['binary']
        self.batch_size = kwargs['batch_size']
        self.res_block = kwargs['res_block']
        self.x_hat_clip = kwargs['x_hat_clip']
        self.epsilon = kwargs['epsilon']
        self.device = kwargs['device']
        self.M = kwargs['M']
        self.MConstrain= kwargs['MConstrain']
        
        # Force the first x_hat to be 0.5
        if self.bias and not self.z2:
            self.register_buffer('x_hat_mask', torch.ones([self.L]))# * 2)) # My code:
            #self.x_hat_mask[0, 0] = 0
            self.x_hat_mask[0] = 0#My code
            self.register_buffer('x_hat_bias', torch.zeros([self.L]))# * 2))#My code
            #self.x_hat_bias[0, 0] = 0.5
            self.x_hat_bias[0] = 0.5#My code

        layers = []
        if self.binary:
            layers.append(
            MaskedLinear(
                self.M,#1,   #My code for one-hot
                self.M if self.net_depth == 1 else self.net_width,#My code for one-hot
                self.n,
                self.bias,
                exclusive=True))
        else:
            layers.append(
            MaskedLinear(
                #self.M,#1,   #My code for one-hot
                1,   #My code for one-hot
                #1 if self.net_depth == 1 else self.net_width,
                1 if self.net_depth == 1 else self.net_width,#My code for one-hot
                #self.M if self.net_depth == 1 else self.net_width,#My code for one-hot
                self.n,
                self.bias,
                exclusive=True))
        #We should change the output feature to 1+2+4?
        
        for count in range(self.net_depth - 2):
            if self.res_block:                              # We didn't use res_block so far
                layers.append(
                    self._build_res_block(self.net_width, self.net_width))
            else:
                layers.append(
                    self._build_simple_block(self.net_width, self.net_width))
        if self.net_depth >= 2:
            #layers.append(self._build_simple_block(self.net_width, 1))
            layers.append(self._build_simple_block(self.net_width, self.M))#My code for one-hot
        #layers.append(nn.Sigmoid())
        #layers.append(nn.log_softmax())#My code for molecular numbers
        self.net = nn.Sequential(*layers)


    def _build_simple_block(self, in_channels, out_channels):
        layers = []
        layers.append(nn.PReLU(in_channels * self.n, init=0.5))
       # layers.append(nn.ReLU())
        layers.append(
            MaskedLinear(
                in_channels, out_channels, self.n, self.bias, exclusive=False))
        block = nn.Sequential(*layers)
        return block

    def _build_res_block(self, in_channels, out_channels):
        layers = []
        layers.append(
            ChannelLinear(in_channels, out_channels, self.n, self.bias))
        layers.append(nn.PReLU(in_channels * self.n, init=0.5))
        #layers.append(nn.ReLU())
        layers.append(
            MaskedLinear(
                in_channels, out_channels, self.n, self.bias, exclusive=False))
        block = ResBlock(nn.Sequential(*layers))
        return block

    def forward(self, x):#,Mask1):
        x = x.view(x.shape[0], -1)
        if self.binary:
            #embedded_x=self.binaryConvert(x.long()) #binary
            embedded_x = F.one_hot(x.to(int), num_classes=self.M).to(default_dtype_torch)#Batchsize X SpinNum X M
            embedded_x=embedded_x.view(x.shape[0], -1)   
        else:
            embedded_x = x.to(int).to(default_dtype_torch)#Batchsize X SpinNum X M
            embedded_x=embedded_x.view(x.shape[0], -1)   
        
        #x_hat = self.net(x) #Use NN to get a probability of the configuration: [Batchsize,SpinNum]
        #x_hat = self.net(embedded_x) #Use NN to get a probability of the configuration: [Batchsize,SpinNum]
        if self.MConstrain[0]==0: #If no number constrain
            x_hat = F.log_softmax(self.net(embedded_x).view(x.shape[0],-1, self.n), dim=1) #My code       
        else:                      #Number constrain
            Temp=self.net(embedded_x).view(x.shape[0],-1, self.n)
            #[Mask1[:,self.MConstrain[ii]:,ii] for ii in arange(self.n)[self.MConstrain<self.M]]=0
            x_hat=torch.ones_like(Temp)*(-100)#zeros_like(Temp)
            index1=arange(self.n)[self.MConstrain<self.M]
            index0=arange(self.n)[self.MConstrain==self.M]
            index11=self.MConstrain[index1[0]]
            index00=self.MConstrain[index0[0]]
            # print(index11,index00)
            # print(index1,index0)
            # ddd
            x_hat[:,:index00,index0] = F.log_softmax(Temp[:,:index00,index0], dim=1) 
            x_hat[:,:index11,index1] = F.log_softmax(Temp[:,:index11,index1], dim=1)
            # print(torch.sum(x_hat[:,:,index0],1))
            # print(torch.sum(x_hat[:,:,index1],1),torch.sum(x_hat[:,:index11,index1],1))
            # dd
            # #print(Temp.shape,Mask1.shape)
            # Mask1=Mask1.repeat(int(Temp.shape[0]/Mask1.shape[0]),1,1)
            # print((Temp*Mask1)[0,:,6:])
            # x_hat = F.softmax(Temp*Mask1, dim=1) # Temp has both negative and positive
            # print(x_hat[0,:,6:])
            
            #x_hat=torch.zeros_like(Temp)
            # bb=torch.cumsum(x_hat,1)
            # print(bb.shape)
            # MConstrain1=torch.tensor(self.MConstrain,dtype=default_dtype_torch,device=self.device).long()-1
            # print(MConstrain1.shape)
            # print(MConstrain1)
            # aa=bb.gather(1, MConstrain1.view(1,-1,1)).view(-1).repeat(self.M,self.n,1).permute(1,2,0)
            # print(aa.shape)
            # x_hat=x_hat/aa
            # print(torch.cumsum(x_hat,1)[0,:,:])
            # #UpBoundary=UpBoundary.view(-1,1).repeat(SampleNeighbor1D.shape[0],1,SampleNeighbor1D.shape[2]) 
            
            # ddd
            # for ii in arange(self.n)[self.MConstrain<self.M]:
            #     #print(self.MConstrain[ii])
            #     x_hat[:,:self.MConstrain[ii],ii] = x_hat/torch.sum(x_hat[:,:self.MConstrain[ii],ii],1)
            #     #x_hat[:,:self.MConstrain[ii],ii] = F.softmax(Temp[:,:self.MConstrain[ii],ii], dim=1) #My code
                #x_hat[:,self.MConstrain[ii]:,ii] = 0 #My code
                #print(x_hat[:,self.MConstrain[ii]:,:] .shape)
        #My code: x_hat is Batchsize X M X SpinNum 
        # print(self.net(embedded_x).shape)
        # print(x_hat.shape)
        # print(x_hat)
        # ddd
        #x_hat = x_hat.view(x_hat.shape[0], 1, self.L)# My code, self.L) 
        if self.x_hat_clip:
            # Clip value and preserve gradient
            with torch.no_grad():
                delta_x_hat = torch.clamp(x_hat, self.x_hat_clip,
                                          1 - self.x_hat_clip) - x_hat
            assert not delta_x_hat.requires_grad
            x_hat = x_hat + delta_x_hat
            dddd

       
        # print(x)
        # print(embedded_x.shape)
        # print(embedded_x)
        
        # print(x_hat.shape)
        # dd
        
        return x_hat

    # sample = +/-1, +1 = up = white, -1 = down = black
    # sample.dtype == default_dtype_torch
    # x_hat = p(x_{i, j} == +1 | x_{0, 0}, ..., x_{i, j - 1})
    # 0 < x_hat < 1
    # x_hat will not be flipped by z2
    
    
    # def Mask(self, batch_size):
    #     Mask1=torch.ones(
    #         [batch_size, self.M, self.L],# My code, self.L],
    #         dtype=default_dtype_torch,
    #         device=self.device)
    #     for ii in arange(self.n)[self.MConstrain<self.M]:
    #         Mask1[:,self.MConstrain[ii]:,ii] = -100#0
    #     #print(Mask1)
    #     return Mask1
        
    def sample(self, batch_size):
        sample = torch.zeros(
            [batch_size, 1, self.L],# My code, self.L],
            dtype=default_dtype_torch,
            device=self.device)
        #Mask1=torch.ones_like(Temp)
        #Mask1=self.Mask(batch_size)
        for i in range(self.L):
            #for j in range(self.L):# My code
                x_hat = self.forward(sample)
                if self.MConstrain[0]==0:
                    sample[:, :, i] = torch.multinomial(
                    torch.exp(x_hat[:, :, i]), 1).to(default_dtype_torch)  # My code                         
                else:
                    sample[:, :, i] = torch.multinomial(
                    torch.exp(x_hat[:, :self.MConstrain[i], i]), 1).to(default_dtype_torch)  # My code                         
                
        # print(sample)
        # print(sample.shape)
        # dd
        
        if self.z2:
            # Binary random int 0/1
           # flip = torch.randint(
              #  2, [batch_size, 1, 1, 1],dtype=sample.dtype, device=sample.device) * 2 - 1
            flip = torch.randint(
                2, [batch_size, 1, 1],dtype=sample.dtype, device=sample.device) * 2 - 1# My code
            sample *= flip

        return sample, x_hat

    def _log_prob(self, sample, x_hat):
        #mask = (sample + 1) / 2 #Spin state of the samples in 1, 0: [Batchsize,1,SpinNum]
        #x_hat is the conditional probability for the current spin up: [Batchsize,1,SpinNum]
        # log_prob = (torch.log(x_hat + self.epsilon) * mask +
        #             torch.log(1 - x_hat + self.epsilon) * (1 - mask))
        #Check the probability data dimension: [Batchsize,1,SpinNum]
        log_prob = torch.zeros_like(sample)#[Batchsize,1,SpinNum]
        for i in range(self.L):
            ids = sample[:,0,i].long()
            log_prob[:,0, i] = x_hat[:,:,i].gather(1, ids.view(-1,1))[:,0]
            
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1) # It seems to sum up all the log(conditional P)
        #Check the probability data dimension: [Batchsize]
        return log_prob

    def log_prob(self, sample):
        #Mask1=self.Mask(self.batch_size)
        x_hat = self.forward(sample) 
        log_prob = self._log_prob(sample, x_hat)

        if self.z2:
            # Density estimation on inverted sample
            sample_inv = -sample
            x_hat_inv = self.forward(sample_inv)
            log_prob_inv = self._log_prob(sample_inv, x_hat_inv)
            log_prob = torch.logsumexp(
                torch.stack([log_prob, log_prob_inv]), dim=0)
            log_prob = log_prob - log(2)

        return log_prob
    
    
   