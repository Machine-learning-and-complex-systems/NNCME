# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 17:35:58 2022

@author: james
"""

import numpy as np
import torch

class AFL:
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
        
    def MaskAFL(self,SampleNeighbor1D_Win,WinProd):
        Mask1=torch.ones_like(WinProd)
        Gu=SampleNeighbor1D_Win[:,0,:] #Gu for different reactions
        Gb=1-Gu
        Mask1[Gu[:,1]!=1,1]=0
        Mask1[Gu[:,2]!=1,2]=0
        Mask1[Gb[:,0]!=1,0]=0
        Mask1[Gb[:,3]!=1,3]=0
        return Mask1
    
    def Propensity(self,Win,Wout,cc,SampleNeighbor1D_Win,SampleNeighbor1D,NotHappen_in_low,NotHappen_in_up,NotHappen_out_low,NotHappen_out_up):
        WinProd=torch.prod(Win,1)
        Mask1=self.MaskAFL(SampleNeighbor1D_Win,WinProd)
        Propensity_in=WinProd*Mask1*cc
        WoutProd=torch.prod(Wout,1)
        Mask1=self.MaskAFL(SampleNeighbor1D,WoutProd)
        Propensity_out=WoutProd*Mask1*cc
        
        return Propensity_in,Propensity_out
    
        
    
    def rates(self): 
    
        
        self.L=2#10#10#16 # Lattice size: 1D  
        initialD=np.array([0,0]).reshape(1,self.L)#0.1#0.1 # the parameter for the initial Poisson distribution
        r=torch.zeros(5) #Reaction rates
        #MConstrain=np.ones(self.L, dtype=int)*self.M
        MConstrain=np.array([2,self.M], dtype=int) #Number constrain
        #MConstrain=np.zeros(1,dtype=int)
        conservation=np.ones(1,dtype=int)
    
        print(MConstrain)
        #Para set 1:
        if self.Para==1:
            sigma_u =0.94
            sigma_b = 0.01
            rho_u = 8.40
            rho_b = 28.1
        # #Para set 2:
        if self.Para==2:
            sigma_u =0.69
            sigma_b = 0.07
            rho_u = 7.2
            rho_b = 40.6
        # #Para set 3:
        if self.Para==3:
            sigma_u =0.44
            sigma_b = 0.08
            rho_u = 0.94
            rho_b = 53.1
        r[0] = sigma_u
        r[1] = sigma_b
        r[2] = rho_u
        r[3] = rho_b
        r[4] = 1
        print(self.IniDistri)
        # Reaction matrix
        ReactionMatLeft=torch.as_tensor([(0, 1,1,0,0), (0,1,0,0,1)]).to(self.device)#SpeciesXReactions
        ReactionMatRight=torch.as_tensor([(1, 0,1,0,0), (1,0,1,1,0)]).to(self.device)#SpeciesXReactions
        IniDistri='delta'
            
        return IniDistri,initialD,r,ReactionMatLeft,ReactionMatRight,MConstrain,conservation

