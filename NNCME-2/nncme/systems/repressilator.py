"""System definition for repressilator."""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 17:35:58 2022

@author: james
"""

import numpy as np
import torch

class repressilator:

    '''Chemical system class for repressilator.'''
    def __init__(self, *args, **kwargs):
        super().__init__()
        #self.n = kwargs['n']
        self.L = kwargs['L']
        self.M = kwargs['M']
        self.bits = kwargs['bits']  
        self.device = kwargs['device']
        #MConstrain = kwargs['MConstrain']
        self.Para = kwargs['Para']
        self.IniDistri = kwargs['IniDistri']
        self.binary = kwargs['binary']
        
    
    def Propensity(self,Win,Wout,cc,SampleNeighbor1D_Win,SampleNeighbor1D,NotHappen_in_low,NotHappen_in_up,NotHappen_out_low,NotHappen_out_up):
        WinProd=torch.prod(Win,1)
        Mask1,Mask2=self.MaskFunction(SampleNeighbor1D_Win,WinProd)
        Propensity_in=WinProd*(Mask1*cc[:,0]+Mask2*cc[:,1])
        WoutProd=torch.prod(Wout,1)
        Mask1,Mask2=self.MaskFunction(SampleNeighbor1D,WoutProd)
        Propensity_out=WoutProd*(Mask1*cc[:,0]+Mask2*cc[:,1])
        
        return Propensity_in,Propensity_out
    
    
    
    def MaskFunction(self,SampleNeighbor1D_Win,WinProd):
        Mask1=torch.ones_like(WinProd)
        Mask2=torch.ones_like(WinProd)
        n1=SampleNeighbor1D_Win[:,6,:]
        n2=SampleNeighbor1D_Win[:,7,:]
        n3=SampleNeighbor1D_Win[:,8,:]
        Mask1[n1[:,1]!=0,1]=0
        Mask1[n1[:,16]>=2,16]=0
        Mask1[n1[:,19]!=1,19]=0
        Mask2[n1[:,1]<=0,1]=0
        Mask2[n1[:,16]>=2,16]=0
        Mask2[n1[:,19]!=2,19]=0
        #n2
        Mask1[n2[:,2]!=0,2]=0
        Mask1[n2[:,17]>=2,17]=0
        Mask1[n2[:,20]!=1,20]=0
        Mask2[n2[:,2]<=0,2]=0
        Mask2[n2[:,17]>=2,17]=0
        Mask2[n2[:,20]!=2,20]=0
        #n3
        Mask1[n3[:,0]!=0,0]=0
        Mask1[n3[:,15]>=2,15]=0
        Mask1[n3[:,18]!=1,18]=0
        Mask2[n3[:,0]<=0,0]=0
        Mask2[n3[:,15]>=2,15]=0
        Mask2[n3[:,18]!=2,18]=0
        return Mask1,Mask2
    
    
    def rates(self):

        self.L=9#10#10#16 # Lattice size: 1D  
        r=torch.zeros(21,2) #Reaction rates   
        # Parameter values
        kmu = 0.5 *self.Para
        kmo = 5e-4 *self.Para
        kp = 0.167 
        gamma_m = 0.005776 *self.Para
        gamma_p = 0.001155 *self.Para#*10 #0.001155
        kr = 1.0 *self.Para
        ku1 = 224.0
        ku2 = 9.0
        
        MConstrain=np.array([self.M,self.M,self.M,self.M,self.M,self.M,3,3,3], dtype=int) #Number constrain
        print(MConstrain)
        
        r[0,0] =kmu #c1
        r[1,0] =kmu #c2
        r[2,0] = kmu #c3
        r[3,0] =kp /2#c1
        r[4,0] =kp /2#c2
        r[5,0] = kp /2#c3
        r[6,0] =gamma_m/2 #c1
        r[7,0] =gamma_m /2#c2
        r[8,0] =gamma_m /2#c1
        r[9,0] =gamma_p /2#c2
        r[10,0] = gamma_p/2 #c3
        r[11,0] =gamma_p /2#c1
        r[12,0] =gamma_p/2 #c2
        r[13,0] = gamma_p/2 #c3
        r[14,0] =gamma_p /2#c1
        r[15,0] =kr /2#c2
        r[16,0] =kr/2 #c1
        r[17,0] =kr /2#c2
        r[18,0] = ku1 #c3
        r[19,0] =ku1 #c1
        r[20,0] =ku1 #c2
        r[0,1] =kmo #c1
        r[1,1] =kmo #c2
        r[2,1] = kmo #c3
        r[3,1] =kp /2#c1
        r[4,1] =kp /2#c2
        r[5,1] = kp /2#c3
        r[6,1] =gamma_m/2 #c1
        r[7,1] =gamma_m /2#c2
        r[8,1] =gamma_m /2#c1
        r[9,1] =gamma_p /2#c2
        r[10,1] = gamma_p/2 #c3
        r[11,1] =gamma_p /2#c1
        r[12,1] =gamma_p/2 #c2
        r[13,1] = gamma_p/2 #c3
        r[14,1] =gamma_p /2#c1
        r[15,1] =kr /2#c2
        r[16,1] =kr /2#c1
        r[17,1] =kr /2#c2
        r[18,1] =ku2#2*ku2 #c3
        r[19,1] =ku2#2*ku2 #c1
        r[20,1] =ku2#2*ku2 #c2
        #self.initialD=np.array([10, 10, 10, 10, 10, 10, -1, -1, -1]).reshape(1,self.L) #for zipf to realize delta
        # self.initialD=np.array([9, 9, 9, 9, 9, 9, -1, -1, -1]).reshape(1,self.L) #for zipf to realize delta
        #self.initialD=np.array([9, 9, 9, 9, 9, 9, 1,1, 1]).reshape(1,self.L) #for zipf to realize delta
        initialD=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(1,self.L) #for zipf to realize delta
        # Reaction matrix #SpeciesXReactions
        # V0=torch.tensor([
        # ( 1,  0,  0,  0,  0,  0,  0,  0,  0), # 0
        # ( 0,  0,  1,  0,  0,  0,  0,  0,  0), # 1
        # ( 0,  0,  0,  0,  1,  0,  0,  0,  0), # 2
        # ( 0,  1,  0,  0,  0,  0,  0,  0,  0), # 3
        # ( 0,  0,  0,  1,  0,  0,  0,  0,  0), # 4
        # ( 0,  0,  0,  0,  0,  1,  0,  0,  0), # 5
        # (-1,  0,  0,  0,  0,  0,  0,  0,  0), # 6
        # ( 0,  0, -1,  0,  0,  0,  0,  0,  0), # 7
        # ( 0,  0,  0,  0, -1,  0,  0,  0,  0), # 8
        # ( 0, -1,  0,  0,  0,  0,  0,  0,  0), # 9
        # ( 0,  0,  0, -1,  0,  0,  0,  0,  0), # 10
        # ( 0,  0,  0,  0,  0, -1,  0,  0,  0), # 11
        # ( 0,  0,  0,  0,  0,  0, -1,  0,  0), # 12
        # ( 0,  0,  0,  0,  0,  0,  0, -1,  0), # 13
        # ( 0,  0,  0,  0,  0,  0,  0,  0, -1), # 14
        # ( 0,  0,  0,  0,  0, -1,  0,  0,  1), # 15
        # ( 0, -1,  0,  0,  0,  0,  1,  0,  0), # 16
        # ( 0,  0,  0, -1,  0,  0,  0,  1,  0), # 17
        # ( 0,  0,  0,  0,  0,  1,  0,  0, -1), # 18
        # ( 0,  1,  0,  0,  0,  0, -1,  0,  0), # 19
        # ( 0,  0,  0,  1,  0,  0,  0, -1,  0), # 20
        # ]).to(self.device)
        
        ReactionMatLeft=torch.as_tensor([
        ( 0,  0,  0,  0,  0,  0,  0,  0,  0), # 0
        ( 0,  0,  0,  0,  0,  0,  0,  0,  0), # 1
        ( 0,  0,  0,  0,  0,  0,  0,  0,  0), # 2
        ( 1,  0,  0,  0,  0,  0,  0,  0,  0), # 3
        ( 0,  0,  1,  0,  0,  0,  0,  0,  0), # 4
        ( 0,  0,  0,  0,  1,  0,  0,  0,  0), # 5
        (1,  0,  0,  0,  0,  0,  0,  0,  0), # 6
        ( 0,  0, 1,  0,  0,  0,  0,  0,  0), # 7
        ( 0,  0,  0,  0, 1,  0,  0,  0,  0), # 8
        ( 0, 1,  0,  0,  0,  0,  0,  0,  0), # 9
        ( 0,  0,  0, 1,  0,  0,  0,  0,  0), # 10
        ( 0,  0,  0,  0,  0, 1,  0,  0,  0), # 11
        ( 0,  0,  0,  0,  0,  0, 1,  0,  0), # 12
        ( 0,  0,  0,  0,  0,  0,  0, 1,  0), # 13
        ( 0,  0,  0,  0,  0,  0,  0,  0, 1), # 14
        ( 0,  0,  0,  0,  0, 1,  0,  0,  0), # 15
        ( 0, 1,  0,  0,  0,  0,  0,  0,  0), # 16
        ( 0,  0,  0, 1,  0,  0,  0,  0,  0), # 17
        ( 0,  0,  0,  0,  0,  0,  0,  0, 1), # 18
        ( 0,  0,  0,  0,  0,  0, 1,  0,  0), # 19
        ( 0,  0,  0,  0,  0,  0,  0, 1,  0), # 20
        ]).to(self.device).t()
        
        ReactionMatRight=torch.as_tensor([
        ( 1,  0,  0,  0,  0,  0,  0,  0,  0), # 0
        ( 0,  0,  1,  0,  0,  0,  0,  0,  0), # 1
        ( 0,  0,  0,  0,  1,  0,  0,  0,  0), # 2
        ( 1,  1,  0,  0,  0,  0,  0,  0,  0), # 3
        ( 0,  0,  1,  1,  0,  0,  0,  0,  0), # 4
        ( 0,  0,  0,  0,  1,  1,  0,  0,  0), # 5
        (0,  0,  0,  0,  0,  0,  0,  0,  0), # 6
        ( 0,  0, 0,  0,  0,  0,  0,  0,  0), # 7
        ( 0,  0,  0,  0, 0,  0,  0,  0,  0), # 8
        ( 0, 0,  0,  0,  0,  0,  0,  0,  0), # 9
        ( 0,  0,  0, 0,  0,  0,  0,  0,  0), # 10
        ( 0,  0,  0,  0,  0, 0,  0,  0,  0), # 11
        ( 0,  0,  0,  0,  0,  0, 0,  0,  0), # 12
        ( 0,  0,  0,  0,  0,  0,  0, 0,  0), # 13
        ( 0,  0,  0,  0,  0,  0,  0,  0, 0), # 14
        ( 0,  0,  0,  0,  0, 0,  0,  0,  1), # 15
        ( 0, 0,  0,  0,  0,  0,  1,  0,  0), # 16
        ( 0,  0,  0, 0,  0,  0,  0,  1,  0), # 17
        ( 0,  0,  0,  0,  0,  1,  0,  0, 0), # 18
        ( 0,  1,  0,  0,  0,  0, 0,  0,  0), # 19
        ( 0,  0,  0,  1,  0,  0,  0, 0,  0), # 20
        ]).to(self.device).t()
        
        IniDistri='delta'
        conservation=np.ones(1,dtype=int)
        
        return IniDistri,initialD,r,ReactionMatLeft,ReactionMatRight,MConstrain,conservation
    
