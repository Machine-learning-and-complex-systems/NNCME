"""System definition for schlogl."""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 17:35:58 2022

@author: james
"""

import numpy as np
import torch

class Schlogl:

    '''Chemical system class for schlogl.'''
    def __init__(self, *args, **kwargs):

        super().__init__()
        #self.n = kwargs['n']
        self.L = kwargs['L']
        self.Sites = kwargs['Sites']
        self.M = kwargs['M']
        self.bits = kwargs['bits']  
        self.device = kwargs['device']
        self.MConstrain = kwargs['MConstrain']
        self.Para = kwargs['Para']
        self.IniDistri = kwargs['IniDistri']
        self.binary = kwargs['binary']
        self.absorb_state = kwargs['absorb_state']
        
    def Propensity(self, Win, Wout, cc, SampleNeighbor1D_Win, SampleNeighbor1D, NotHappen_in_low, NotHappen_in_up, NotHappen_out_low, NotHappen_out_up):
        Propensity_in = torch.prod(Win, 1) * cc
        Propensity_out = torch.prod(Wout, 1) * cc
        return Propensity_in, Propensity_out
    
    def rates(self): 

        L=1
        K=4
        Sites=self.Sites
        L_total=L*Sites
        K_total=K*Sites
        
        #self.L=3#10#10#16 # Lattice size: 1D  
        r=torch.zeros(4) #Reaction rates   
        # Parameter values
        na=1
        nb=self.Para#1
        
        c1, c2, c3, c4=2.676,0.040, 108.102, 37.881
        d=8.2207
        
        # V = 25
        # na = 1.0
        # nb = 2.0
        # c1 = 3.0/V
        # c2 = 0.6/V/V
        # c3 = 0.25*V
        # c4 = 2.95

        r[0] = c1*na
        r[1] = c2
        r[2] = c3*nb
        r[3] = c4

        r=r.repeat(self.Sites)
        
        if d!=0:
            aa=d*torch.ones(1+(self.Sites-2)*2+1)
            r=torch.cat((r,aa))

        D=50
        IniDistri='delta'
        initialD=np.tile(D,Sites).reshape(1,self.L)#np.array([D]).reshape(1,int(self.L/self.Sites))#0.1#0.1 # the parameter for the initial Poisson distribution
        
        # X A B
        Left=np.array([2,3,0,1])#,[0,0,1,0],[1,0,0,0]])
        Right=np.array([3,2,1,0])#,[0,0,0,1],[0,1,0,0]])
        # Whole=Right-Left
        update1L=np.zeros((L_total,K_total),dtype=int)
        update1R=np.zeros((L_total,K_total),dtype=int)
        for i in range(Sites):
            update1L[i*L:(i+1)*L,i*K:(i+1)*K]=Left
            update1R[i*L:(i+1)*L,i*K:(i+1)*K]=Right
        update1=update1R-update1L
        if Sites>1 and d!=0:
            update2L=np.zeros((L_total,1+(Sites-2)*2+1),dtype=int)
            update2R=np.zeros((L_total,1+(Sites-2)*2+1),dtype=int)
            update2L[0,0]=1 #X1->X2
            update2L[-L,-1]=1 #Xlast->Xlast-1
            update2R[L,0]=1 #X1->X2
            update2R[-L*2,-1]=1 #Xlast->Xlast-1
            for i in range(1,Sites-1):
                update2L[i*L,i*2-1]=1
                update2L[i*L,i*2]=1
                update2R[(i-1)*L,i*2-1]=1
                update2R[(i+1)*L,i*2]=1
            # update2=update2R-update2L
            updateL=np.concatenate((update1L,update2L),axis=1)
            updateR=np.concatenate((update1R,update2R),axis=1)
            # Schlogl_update=update.T
            
            ReactionMatLeftTotal=torch.tensor(updateL).to(self.device)#torch.cat((ReactionMatLeftTotal_1,ReactionMatLeftTotal_2),axis=1)
            ReactionMatRightTotal=torch.tensor(updateR).to(self.device)#torch.cat((ReactionMatRightTotal_1,ReactionMatRightTotal_2),axis=1)
        else:
            ReactionMatLeftTotal=torch.tensor(update1L).to(self.device)#torch.cat((ReactionMatLeftTotal_1,ReactionMatLeftTotal_2),axis=1)
            ReactionMatRightTotal=torch.tensor(update1R).to(self.device)#torch.cat((ReactionMatRightTotal_1,ReactionMatRightTotal_2),axis=1)
        MConstrain=np.zeros(1,dtype=int)
        conservation=np.ones(1,dtype=int)
        if self.binary:
            self.bits=int(np.ceil(np.log2(conservation)))
        
        print(conservation,MConstrain)
            
        return IniDistri,initialD,r,ReactionMatLeftTotal,ReactionMatRightTotal,MConstrain,conservation
    
