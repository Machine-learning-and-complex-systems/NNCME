# GRU for VAN

import numpy as np
import torch
from numpy import log
from torch import nn

from utils import default_dtype_torch
import torch.nn.functional as F
#torch.autograd.set_detect_anomaly(True)

class GRU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        #self.n = kwargs['n']
        self.L = kwargs['L']
        self.M = kwargs['M']
        self.bits = kwargs['bits']
        self.n = self.L#**2  # My code: Number of sites
        self.net_depth = kwargs['net_depth']
        self.net_width = kwargs['net_width'] # hidden neurons
        self.bias = kwargs['bias']
        # self.z2 = kwargs['z2']
        self.res_block = kwargs['res_block']
        # self.x_hat_clip = kwargs['x_hat_clip']
        self.epsilon = kwargs['epsilon']
        self.device = kwargs['device']
        self.reverse = kwargs['reverse']
        self.binary = kwargs['binary']
        self.conservation = torch.as_tensor(kwargs['conservation']).to(self.device)
        self.MConstrain= kwargs['MConstrain']

        self.order = list(range(self.n))
        
        # model parameters
        if self.binary:
            if self.conservation>1:
              for i in range(self.net_depth):
                if i==0:
                    self.rnn = nn.GRUCell(self.bits, self.net_width)
                    self.fc = nn.Linear(self.net_width, (self.conservation+1).detach().cpu().numpy())  
            else:
              for i in range(self.net_depth):
                if i==0:
                    self.rnn = nn.GRUCell(self.bits, self.net_width)
                    self.fc = nn.Linear(self.net_width, self.M)
                if i==1:
                    self.rnn1 = nn.GRUCell(self.M, self.net_width)
                    self.fc1 = nn.Linear(self.net_width, self.M)
                if i==2:
                    self.rnn2 = nn.GRUCell(self.M, self.net_width)
                    self.fc2 = nn.Linear(self.net_width, self.M)
                if i==3:
                    self.rnn3 = nn.GRUCell(self.M, self.net_width)
                    self.fc3 = nn.Linear(self.net_width, self.M)
        else:
            if self.conservation>1:
              for i in range(self.net_depth):
                if i==0:
                    self.rnn = nn.GRUCell(1, self.net_width)
                    self.fc = nn.Linear(self.net_width, (self.conservation+1).detach().cpu().numpy())  
            else: 
              for i in range(self.net_depth):
                if i==0:
                    #self.rnn = nn.GRUCell(self.M, self.net_width)
                    self.rnn = nn.GRUCell(1, self.net_width)
                    self.fc = nn.Linear(self.net_width, self.M)
                if i==1:
                    self.rnn1 = nn.GRUCell(self.M, self.net_width)
                    self.fc1 = nn.Linear(self.net_width, self.M)
                if i==2:
                    self.rnn2 = nn.GRUCell(self.M, self.net_width)
                    self.fc2 = nn.Linear(self.net_width, self.M)
                if i==3:
                    self.rnn3 = nn.GRUCell(self.M, self.net_width)
                    self.fc3 = nn.Linear(self.net_width, self.M)
        

    def binaryConvert(self,x):
        mask = 2**torch.arange(self.bits).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
    
    def _forward(self, x, h,ii):
        batch_size = x.shape[0]
        
        if self.binary:
            embedded_x=self.binaryConvert(x.long()) #binary
        else:
            #embedded_x = F.one_hot(x.long(), num_classes=self.M)#targets_to_one_hot
            embedded_x=x.long().reshape(-1,1) #      
        if self.net_depth==1:
            h_layer = self.rnn(embedded_x.to(default_dtype_torch), h)
            if self.MConstrain[0]==0: #If no number constrain
                y = F.log_softmax(self.fc(h_layer), dim=1)
            else:                      #Number constrain
                y = F.log_softmax(self.fc(h_layer)[:,:self.MConstrain[ii]], dim=1)
        else:
            for i in range(self.net_depth):
                if i==0:
                    Temp = self.rnn(embedded_x.to(default_dtype_torch), h[:,:,i])
                    h_layer=Temp.view(batch_size,self.net_width,1)
                if i==1:
                    Temp = self.rnn1(self.fc1(Temp), h[:,:,i])
                    h_layer=torch.cat((h_layer,Temp.view(batch_size,self.net_width,1)),2)
                if i==2:
                    Temp = self.rnn2(self.fc2(Temp), h[:,:,i])
                    h_layer=torch.cat((h_layer,Temp.view(batch_size,self.net_width,1)),2)
                if i==3:
                    Temp = self.rnn3(self.fc3(Temp), h[:,:,i])
                    h_layer=torch.cat((h_layer,Temp.view(batch_size,self.net_width,1)),2)
                    
            if self.MConstrain[0]==0: #If no number constrain
                y = F.log_softmax(self.fc(h_layer[:,:,-1]), dim=1)
            else:                      #Number constrain
                y = F.log_softmax(self.fc(h_layer[:,:,-1])[:,:self.MConstrain[ii]], dim=1)

        return h_layer, y
    
    def _forward_conserve(self, x, h,remain):
        if self.binary:
            embedded_x=self.binaryConvert(x.long()) #binary
        else:
            #embedded_x = F.one_hot(x.long(), num_classes=self.M)#targets_to_one_hot
            embedded_x=x.long().reshape(-1,1) #      

        if self.net_depth==1:    
            h_layer = self.rnn(embedded_x.to(default_dtype_torch), h)
            expy = F.softmax(self.fc(h_layer), dim=1)
            temp=torch.cumsum(expy, dim=1)
            normalizationfactor=torch.gather(temp,1,remain.view(-1,1).abs()) #If negative index, still make this normalization, but in logprob does not take them
            expy=expy/normalizationfactor
            Aux=torch.arange(0, self.conservation+1).repeat(remain.shape[0],1).to(self.device,dtype=default_dtype_torch)
            remain2=remain.repeat(self.conservation+1,1).permute(1,0).to(self.device,dtype=default_dtype_torch)
            expy[remain2<Aux]= torch.as_tensor(self.epsilon).to(self.device,dtype=default_dtype_torch)
            # ###Method 2 by for loop
            # for i in range(expy.shape[0]):
            #     expy[i, remain[i]+1:] = self.epsilon
            ## try to not use for loop, and redo renormalization if still use for loop
            ### Need to check the sample procedure to sample configurations with conserved quantities
            ### Then test the new non-binary encoding's accuracy for the homo1 example
            y=torch.log(expy)        
        return h_layer, y

    
    def log_prob(self, x):
        batch_size = x.shape[0]
        log_prob = torch.zeros_like(x)
        x_init = torch.zeros(batch_size, dtype=default_dtype_torch, device=self.device)
        if self.net_depth==1:
            h_init = torch.zeros(batch_size, self.net_width,dtype=default_dtype_torch, device=self.device)
        else:
            h_init = torch.zeros(batch_size, self.net_width, self.net_depth,dtype=default_dtype_torch, device=self.device)
        epsilon=torch.log(torch.as_tensor(self.epsilon).to(self.device,dtype=default_dtype_torch))#-torch.as_tensor(30).to(self.device,dtype=default_dtype_torch)
        
        i=0
        
        if self.conservation==1:
            h, y= self._forward(x_init, h_init,i)
            ids = x[:,0].long() 
            log_prob[:, 0] = y.gather(1, ids.view(-1,1))[:,0]#
        if self.conservation>1:
            remain=(self.conservation-x_init.detach()).long()
            h, y = self._forward_conserve(x_init, h_init,remain)
            ids = x[:,0].long()#x[:,i-1].long()#NewNew
            log_probTemp= y.gather(1, ids.view(-1,1))[:,0]# #renormalized prob for those small number of this species to keep total number conservation
            Mask0=remain<0 # drop
            Mask1=remain>=0 #keep
            log_prob[Mask1, i]=log_probTemp[Mask1]
            log_prob[Mask0, i-1:self.n]=epsilon
        
        if self.reverse:
            log_prob_rev = torch.zeros_like(x)
            h_rev, y_rev= self._forward(x_init, h_init,i)
            ids_rev = x[:,-1].long()
            log_prob_rev[:, 0] = y_rev.gather(1, ids_rev.view(-1,1))[:,0]#y[:, 0] * mask[:, 0] + y[:, 1] * (1.0 - mask[:, 0])
        
        
                        
        for i in range(1, self.n):
            if self.conservation==1:
                h, y = self._forward(x[:, i - 1], h,i)
                ids = x[:,i].long()#x[:,i-1].long()#NewNew
                log_prob[:, i] = y.gather(1, ids.view(-1,1))[:,0]

            if self.conservation>1:
                remain=(self.conservation-torch.sum(x[:,:i],1).detach()).long()
                ###remain is correct: use this for sample and logp, 
                ###from the zero site with remain=self.conservation-0? 
                ###yes, because its prob also needs to be normalized.
                h, y1 = self._forward_conserve(x[:, i - 1], h,remain)
                ids = x[:,i].long()#x[:,i-1].long()#NewNew
                ## Method1: when there is mask in the y1, logp     
                log_prob[:, i] = y1.gather(1, ids.view(-1,1))[:,0]
                # ## Method2: when there is no mask in the y1, logp    
                # log_probTemp= y1.gather(1, ids.view(-1,1))[:,0] #renormalized prob for those small number of this species to keep total number conservation
                # Mask0=remain<0 # drop
                # Mask1=remain>=0 #keep
                # log_prob[Mask1, i]=log_probTemp[Mask1]
                # log_prob[Mask0, i-1:self.n]=epsilon
                
                #Revise the previous conditional prob is the number up to this site> conservation number
                if i==self.n-1:
                    Mask=remain-ids==0 #The configuration preserving conservation:Make its False prob to be zero in the end
                    log_prob[Mask, i]=0#torch.as_tensor(0).to(self.device)#*log_prob[Mask, i]
                    log_prob[~Mask, i]=epsilon#-torch.as_tensor(100).to(self.device)#-30*torch.ones(1).to(self.device)

            if self.reverse:
                h_rev, y_rev = self._forward(x[:, self.n-i ], h_rev,i)
                ids_rev = x[:,self.n-i-1].long()#x[:,self.n-i].long()#NewNew
                log_prob_rev[:, i] = y_rev.gather(1, ids_rev.view(-1,1))[:,0]
        
        log_probsum=log_prob.sum(dim=1)
        if self.reverse:
            log_probsum=torch.logsumexp(torch.stack([log_probsum,log_prob_rev.sum(dim=1)]), dim=0)- log(2)
       
        return log_probsum
    
    def sample_(self, batch_size):
        samples = torch.zeros([batch_size, self.n],dtype=default_dtype_torch,device=self.device)
        
        x_init = torch.zeros(batch_size, dtype=default_dtype_torch, device=self.device)
        if self.net_depth==1:
            h_init = torch.zeros(batch_size, self.net_width,dtype=default_dtype_torch, device=self.device)
        else:
            h_init = torch.zeros(batch_size, self.net_width, self.net_depth,dtype=default_dtype_torch, device=self.device)
        i=0
        
        if self.conservation==1:
            h, y = self._forward(x_init, h_init,i)
            p = torch.exp(y)#[:, 0]
        if self.conservation>1:
            remain=(self.conservation-x_init.detach()).long()
            h, y = self._forward_conserve(x_init, h_init,remain)
            p = torch.exp(y)
        
        
        #print(p)
        samples[:, 0] = torch.multinomial(p, 1)[:,0]

        
        for i in range(1, self.n):
            if self.conservation==1:
                h, y = self._forward(samples[:, i - 1], h,i)
                p = torch.exp(y)#[:, 0]
                samples[:, i] = torch.multinomial(p, 1)[:,0]
            
            ### code conservation for the sampling procedure: 
            ### p should only has probability for the allowable species number < total conservation            
            if self.conservation>1:
                remain=(self.conservation-torch.sum(samples[:,:i],1).detach()).long()
                if i<self.n-1:
                    h, y = self._forward_conserve(samples[:, i - 1], h,remain)
                    p = torch.exp(y)  
                    Temp= torch.multinomial(p, 1)[:,0]
                    Mask=remain<=0 #The configuration preserving conservation:Make its False prob to be zero in the end
                    samples[Mask, i]=torch.as_tensor(0).to(self.device,dtype=default_dtype_torch)#*log_prob[Mask, i]
                    samples[~Mask, i]=Temp[~Mask].to(default_dtype_torch)#-torch.as_tensor(100).to(self.device)#-30*torch.ones(1).to(self.device)      
                if i==self.n-1:
                    Mask=remain<=0 #The configuration preserving conservation:Make its False prob to be zero in the end
                    samples[Mask, i]=torch.as_tensor(0).to(self.device,dtype=default_dtype_torch)#Temp[Mask].to(default_dtype_torch)#.long()#torch.as_tensor(0).to(self.device)#*log_prob[Mask, i]
                    samples[~Mask, i]=remain[~Mask].to(default_dtype_torch)#.long()#-torch.as_tensor(100).to(self.device)#-30*torch.ones(1).to(self.device)
                #Check: make sure the last-species take the remaining number, 
                #and the total number of each sample is conserved: done, 
                #and the logP from the NN is zero for the species after > conservation: done
    
        return samples
            
    def sample(self, batch_size):
        samples=self.sample_(batch_size)
        x_hat = torch.zeros_like(samples) #my code
        
        return samples,x_hat