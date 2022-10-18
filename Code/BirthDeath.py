# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 17:35:58 2022

@author: james
"""

import numpy as np
import torch


class BirthDeath:
    def __init__(self, *args, **kwargs):
        super().__init__()
        #self.n = kwargs['n']
        self.L = kwargs['L']
        self.M = kwargs['M']
        self.bits = kwargs['bits']  
        self.device = kwargs['device']
        self.MConstrain = kwargs['MConstrain']
        self.Para = kwargs['Para']
        self.IniDistri = kwargs['IniDistri']
        self.binary = kwargs['binary']
        self.order = kwargs['order']
        
    def Propensity(self,Win,Wout,cc,SampleNeighbor1D_Win,SampleNeighbor1D,NotHappen_in_low,NotHappen_in_up,NotHappen_out_low,NotHappen_out_up):
        Propensity_in=torch.prod(Win,1)*cc#torch.tensor(r, dtype=torch.float64).to(self.device)   
        Propensity_out=torch.prod(Wout,1)*cc#torch.tensor(r, dtype=torch.float64).to(self.device)    
    
        return Propensity_in,Propensity_out
    
        
    
    
    def rates(self):  

        IniDistri='poisson'
        self.L=1#10#10#16 # Lattice size: 1D  
        r=torch.zeros(2) #Reaction rates
        r[0] = 0.1 #k2
        r[1] = 0.01 #k1
        initialD=np.array([1]) # the parameter for the initial Poisson distribution
        # Reaction matrix
        ReactionMatLeft=torch.as_tensor([(0, 1)]).to(self.device)#SpeciesXReactions
        ReactionMatRight=torch.as_tensor([(1, 0)]).to(self.device)#SpeciesXReactions
        MConstrain=np.zeros(1,dtype=int)
        conservation=np.ones(1,dtype=int)
        
        return IniDistri,initialD,r,ReactionMatLeft,ReactionMatRight,MConstrain,conservation

