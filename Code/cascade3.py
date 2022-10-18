# -*- coding: utf-8 -*-
import numpy as np
import torch

# ########cascade1:

    
class cascade3:
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
        k=Win.shape[1]-1
        Win[:,k,0]=1+(cc[0,1]/cc[0,0])/(cc[1,1]+SampleNeighbor1D_Win[:,k,0]**cc[2,1])
        Wout[:,k,0]=1+(cc[0,1]/cc[0,0])/(cc[1,1]+SampleNeighbor1D[:,k,0]**cc[2,1])
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
        k = 5
        b=1
        gamma =1
        km=100 #(1+(km/b)x^H/(k0+x^H))
        k0=10
        H=1
        initialD=np.ones((1,self.L))*(-1)#0.1#0.1 # the parameter for the initial Poisson distribution
        r=torch.zeros(2*self.L,2)
        r[0,0] = b
        r[0,1] =km
        r[1,1] =k0
        r[2,1] =H
        for ii in range(self.L): # decay
            r[2*ii+1,0]=gamma
        for ii in range(1,self.L): # decay
            r[2*ii,0]=k
        ReactionMatLeft = torch.zeros((self.L,2*self.L)).to(self.device)
        for i in range(self.L): # decay
            ReactionMatLeft[i,2*i+1]=1
        for i in range(1,self.L): # decay
            ReactionMatLeft[i-1,2*i]=1
        ReactionMatRight = torch.zeros((self.L,2*self.L)).to(self.device)
        ReactionMatRight[0,0]=1
        for i in range(1,self.L): # decay
            ReactionMatRight[i,2*i]=1
        print(ReactionMatRight.shape)
        IniDistri='delta'
        # have checked (ReactionMatRight-ReactionMatLeft)
        MConstrain=np.zeros(1,dtype=int)
        conservation=np.ones(1,dtype=int)
        
        return IniDistri,initialD,r,ReactionMatLeft,ReactionMatRight,MConstrain,conservation