# -*- coding: utf-8 -*-
import numpy as np
import torch
from utils import default_dtype_torch

# ########cascade1:

class cascade2:
    def __init__(self, *args, **kwargs):
        super().__init__()
        #self.n = kwargs['n']
        self.L = kwargs['L']
        self.M = kwargs['M']
        self.bits = kwargs['bits']  
        self.device = kwargs['device']
        self.MConstrain = kwargs['MConstrain']
        self.Para = kwargs['Para']
        #self.initialD = kwargs['initialD']
        

    def Propensity(self,Win,Wout,cc,SampleNeighbor1D_Win,SampleNeighbor1D,NotHappen_in_low,NotHappen_in_up,NotHappen_out_low,NotHappen_out_up):
         # for k in range(Win.shape[1]):
        #     #Win[:,:,1:Win.shape[2]:2]=1+(cc[0,1]/cc[2,0]*SampleNeighbor1D_Win[:,:,1:Win.shape[2]:2]**cc[2,1])/(cc[1,1]+SampleNeighbor1D_Win[:,:,1:Win.shape[2]:2]**cc[2,1])
        #     Win[:,k,2*k+1]=1+(cc[0,1]/cc[2,0]*SampleNeighbor1D_Win[:,k,2*k+1]**cc[2,1])/(cc[1,1]+SampleNeighbor1D_Win[:,k,2*k+1]**cc[2,1])
        #     Wout[:,k,2*k+1]=1+(cc[0,1]/cc[2,0]*SampleNeighbor1D[:,k,2*k+1]**cc[2,1])/(cc[1,1]+SampleNeighbor1D[:,k,2*k+1]**cc[2,1])
        #Standard:
        for k in range(Win.shape[1]-1):
            Win[:,k,2*(k+1)]=1+(cc[0,1]/cc[2,0]*SampleNeighbor1D_Win[:,k,2*(k+1)]**cc[2,1])/(cc[1,1]+SampleNeighbor1D_Win[:,k,2*(k+1)]**cc[2,1])
            Wout[:,k,2*(k+1)]=1+(cc[0,1]/cc[2,0]*SampleNeighbor1D[:,k,2*(k+1)]**cc[2,1])/(cc[1,1]+SampleNeighbor1D[:,k,2*(k+1)]**cc[2,1])
        
        # #Only last-species nonlinear:
        # for k in range(Win.shape[1]-2):
        #     Win[:,k,2*(k+1)]=1
        #     Wout[:,k,2*(k+1)]=1
        # for k in range(Win.shape[1]-2,Win.shape[1]-1):
        #     Win[:,k,2*(k+1)]=1+(cc[0,1]/cc[2,0]*SampleNeighbor1D_Win[:,k,2*(k+1)]**cc[2,1])/(cc[1,1]+SampleNeighbor1D_Win[:,k,2*(k+1)]**cc[2,1])
        #     Wout[:,k,2*(k+1)]=1+(cc[0,1]/cc[2,0]*SampleNeighbor1D[:,k,2*(k+1)]**cc[2,1])/(cc[1,1]+SampleNeighbor1D[:,k,2*(k+1)]**cc[2,1])
        
        Win[NotHappen_in_low]=0 # Make the not-happen reaction with zero chemical species, giving zero flux when multiplied below
        Win[NotHappen_in_up]=0 # Make the not-happen reaction with zero chemical species, giving zero flux when multiplied below
        Wout[NotHappen_out_low]=0 # Make the not-happen reaction with zero chemical species, giving zero flux when multiplied below
        Wout[NotHappen_out_up]=0 # Make the not-happen reaction with zero chemical species, giving zero flux when multiplied below
        WinProd=torch.prod(Win,1)
        Propensity_in=WinProd*cc[:,0]
        WoutProd=torch.prod(Wout,1)
        Propensity_out=WoutProd*cc[:,0]
        
        return Propensity_in,Propensity_out
    
    def rates(self): 
        
        
        t0=25
        NN=1.5*(15/self.L)
        beta =10*(t0/NN)
        gamma =0.1*(t0/1)#1#0.2#1
        b=1*(t0/NN)
        km=100*(t0/NN)#10#100 #(1+(km/b)x^H/(k0+x^H))
        k0=100*(1/NN)#10
        H=1
        # beta =10
        # #k = 5
        # b=1
        # H=1
        # #original
        # gamma =1
        # km=100 #(1+(km/b)x^H/(k0+x^H))
        # k0=10
        ##for L=5
        #gamma =0.2#1
        #km=10#100 #(1+(km/b)x^H/(k0+x^H))
        #k0=10
        # ##for L=15
        # gamma =0.1#1
        # km=100 #(1+(km/b)x^H/(k0+x^H))
        # k0=100     
        if self.Para!=1:
            self.MConstrain=np.ones(self.L, dtype=int)*int(self.Para) #Number constrain
            self.MConstrain[-1]=int(self.M)
        initialD=np.ones((1,self.L))*(-1)#0.1#0.1 # the parameter for the initial Poisson distribution
        r=torch.zeros(2*self.L,2)
        r[0,0] = beta
        r[0,1] =km
        r[1,1] =k0
        r[2,1] =H
        for ii in range(self.L): # decay
            r[2*ii+1,0]=gamma
        for ii in range(1,self.L): # decay
            r[2*ii,0]=b
        ReactionMatLeft = torch.zeros((self.L,2*self.L)).to(self.device)
        for i in range(self.L): # decay
            ReactionMatLeft[i,2*i+1]=1
        for i in range(1,self.L): # decay
            ReactionMatLeft[i-1,2*i]=1
        ReactionMatRight = torch.zeros((self.L,2*self.L)).to(self.device)
        ReactionMatRight[0,0]=1
        for i in range(1,self.L): # decay
            ReactionMatRight[i,2*i]=1
        IniDistri='delta'
        MConstrain=np.zeros(1,dtype=int)
        conservation=np.ones(1,dtype=int)
        
        return IniDistri,initialD,r,ReactionMatLeft,ReactionMatRight,MConstrain,conservation
    
    
