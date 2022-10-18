# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 17:35:58 2022

@author: james
"""

import numpy as np
import torch


class GeneExp:
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
        
        #delta=0.1
        IniDistri='delta'
        self.L=2#10#10#16 # Lattice size: 1D  
        r=torch.zeros(4) #Reaction rates
        # #Parameters 1:
        # r[0] = 0.01 #kr
        # r[1] = 1 #kp
        # r[2] = 0.1 #gamma r
        # r[3] = 0.002 #gamma p  
        # Parameters 2:
        r[0] = 0.1 #kr
        r[1] = 0.1 #kp
        r[2] = 0.1 #yr
        r[3] = 0.002 #yp  
        initialD=np.array([0,0]).reshape(1,self.L)#0.1#0.1 # the parameter for the initial Poisson distribution
        #self.IniDistri='poisson'
        print(self.IniDistri)
        # Reaction matrix
        if self.order==1:
            ReactionMatLeft=torch.as_tensor([(0, 1,1,0), (0,0,0,1)]).to(self.device)#SpeciesXReactions
            ReactionMatRight=torch.as_tensor([(1, 1,0,0), (0,1,0,0)]).to(self.device)#SpeciesXReactions
        
        # # Inversing the order of mRNA and protein, Scenario 2:
        if self.order==2:
            ReactionMatLeft=torch.as_tensor([(0,0,0,1),(0, 1,1,0)]).to(self.device)#SpeciesXReactions
            ReactionMatRight=torch.as_tensor([(0,1,0,0),(1, 1,0,0)]).to(self.device)#SpeciesXReactions
        
        MConstrain=np.zeros(1,dtype=int)
        conservation=np.ones(1,dtype=int)
        
        # Stoichiometric matrix
        #V=ReactionMatRight-ReactionMatLeft #SpeciesXReactions    
        return IniDistri,initialD,r,ReactionMatLeft,ReactionMatRight,MConstrain,conservation


