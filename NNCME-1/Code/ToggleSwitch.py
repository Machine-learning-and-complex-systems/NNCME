# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 17:35:58 2022

@author: james
"""

import numpy as np
import torch

class ToggleSwitch:
    def __init__(self, *args, **kwargs):
        super().__init__()
        #self.n = kwargs['n']
        self.L = kwargs['L']
        self.M = kwargs['M']
        self.bits = kwargs['bits']  
        self.device = kwargs['device']
        self.MConstrain = kwargs['MConstrain']
        # self.Para = kwargs['Para']
        self.IniDistri = kwargs['IniDistri']
    
    # It is used to constrain the count of certain species. 
    # For example, DNA typically only has the count of 0 or 1 inside a cell. 
    # The "Mask" function allows only the reactions with such a proper count to occur.
    def MaskToggleSwitch(self,SampleNeighbor1D_Win,WinProd):
        Mask1=torch.ones_like(WinProd)
        Gx=SampleNeighbor1D_Win[:,0,:] #Gx for different reactions: the second dimension of SampleNeighbor1D_Win is the label of species
        Gy=SampleNeighbor1D_Win[:,1,:] #Gy for different reactions: the second dimension of SampleNeighbor1D_Win is the label of species
        Gx1=1-Gx
        Gy1=1-Gy
        Mask1[Gx[:,0]!=1,0]=0 #The second dimension of Gx and Mask1 is the label of reactions
        Mask1[Gx[:,5]!=1,5]=0
        Mask1[Gy[:,1]!=1,1]=0
        Mask1[Gy[:,4]!=1,4]=0
        Mask1[Gx1[:,7]!=1,7]=0
        Mask1[Gy1[:,6]!=1,6]=0
        
        return Mask1
    
    def Propensity(self,Win,Wout,cc,SampleNeighbor1D_Win,SampleNeighbor1D,NotHappen_in_low,NotHappen_in_up,NotHappen_out_low,NotHappen_out_up):
        WinProd=torch.prod(Win,1)
        Mask1=self.MaskToggleSwitch(SampleNeighbor1D_Win,WinProd)
        Propensity_in=WinProd*Mask1*cc
        WoutProd=torch.prod(Wout,1)
        Mask1=self.MaskToggleSwitch(SampleNeighbor1D,WoutProd)
        Propensity_out=WoutProd*Mask1*cc
        
        return Propensity_in,Propensity_out
    
    
    
    def rates(self): 
    
        
        self.L=4#10#10#16 # Lattice size: 1D  
        initialD=np.array([1,1,0,0]).reshape(1,self.L)#0.1#0.1 # the parameter for the initial Poisson distribution
        IniDistri='delta'
        r=torch.zeros(8) #Reaction rates
        MConstrain=np.array([2,2,self.M,self.M], dtype=int) #Number constrain
        conservation=np.ones(1,dtype=int)

        sx=sy=50
        dx=dy=1
        by=bx=1e-4
        uy=ux=0.1
        
        r[0] = sx
        r[1] = sy
        r[2] = dx
        r[3] = dy
        r[4] = by
        r[5] = bx
        r[6] = uy
        r[7] = ux

        # Reaction matrix
        ReactionMatLeft=torch.as_tensor([(1,0,0,0,0,1,0,0),(0,1,0,0,1,0,0,0),(0,0,1,0,2,0,0,0),(0,0,0,1,0,2,0,0)]).to(self.device)#SpeciesXReactions
        ReactionMatRight=torch.as_tensor([(1,0,0,0,0,0,0,1),(0,1,0,0,0,0,1,0),(1,0,0,0,0,0,2,0),(0,1,0,0,0,0,0,2)]).to(self.device)#SpeciesXReactions
            
        return IniDistri,initialD,r,ReactionMatLeft,ReactionMatRight,MConstrain,conservation

