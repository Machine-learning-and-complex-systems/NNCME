# -*- coding: utf-8 -*-
import numpy as np
import torch

# ########cascade1:

class cascade1:
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
        Propensity_in=torch.prod(Win,1)*cc#torch.tensor(r, dtype=torch.float64).to(args.device)   
        Propensity_out=torch.prod(Wout,1)*cc#torch.tensor(r, dtype=torch.float64).to(args.device) 
        
        return Propensity_in,Propensity_out
    
    def rates(self): 
        
        beta =10
        k = 5
        gamma =1
        initialD=np.ones((1,self.L))*(-1)#0.1#0.1 # the parameter for the initial Poisson distribution
        r=torch.zeros(2*self.L)
        r[0] = beta
        for ii in range(self.L): # decay
            r[2*ii+1]=gamma
        for ii in range(1,self.L): # decay
            r[2*ii]=k
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
        # have checked (ReactionMatRight-ReactionMatLeft)
        IniDistri='delta'
        MConstrain=np.zeros(1,dtype=int)
        conservation=np.ones(1,dtype=int)
        
        return IniDistri,initialD,r,ReactionMatLeft,ReactionMatRight,MConstrain,conservation


