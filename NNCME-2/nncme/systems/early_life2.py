"""System definition for early life2."""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 17:35:58 2022

@author: james
"""

import numpy as np
import torch

class EarlyLife2:

    '''Chemical system class for early life2.'''
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
        
    def Propensity(self,Win,Wout,cc,SampleNeighbor1D_Win,SampleNeighbor1D,NotHappen_in_low,NotHappen_in_up,NotHappen_out_low,NotHappen_out_up):
        Propensity_in=torch.prod(Win,1)*cc#torch.tensor(r, dtype=torch.float64).to(args.device)   
        Propensity_out=torch.prod(Wout,1)*cc#torch.tensor(r, dtype=torch.float64).to(args.device)    
    
        return Propensity_in,Propensity_out
    
        
    
    
    def rates(self):
        r=torch.zeros(6) #Reaction rates   
        # Parameter values
        ka = 1#1
        kn = 1
        kd = 1
        #self.Para=0.1
        V = self.Para#0.1#1#10
        print(V)
        # D=int(self.M/3-1)
        D=int((self.M-2)/2)
        r[0] = ka/V #kr
        r[1] = ka/V #kp
        r[2] = kn #yr
        r[3] = kn #yp  
        r[4] = kd #yr
        r[5] = kd #yp  
        delta=kd/self.M # Diffusion strength at the intermediate transition 1e-1#1e-3
        r=r.repeat(self.Sites)
        
        aa=delta*torch.ones(2+(self.Sites-2)*4+2)
        r=torch.cat((r,aa))

        
        initialD=np.array([2,D,D]).reshape(1,int(self.L/self.Sites))#0.1#0.1 # the parameter for the initial Poisson distribution
        initialD=np.repeat(initialD, self.Sites,axis=0).reshape(1,self.L)
        # Reaction matrix: wrong, need matrix block
        # ReactionMatLeft=torch.as_tensor([(1, 1,1,1,0,0,0,0,0,0,0,0,0,0), (1,0,0,0,1,0,0,0,0,0,0,0,0,0), (0,1,0,0,0,1,0,0,0,0,0,0,0,0)]).to(self.device)#SpeciesXReactions
        # ReactionMatRight=torch.as_tensor([(0, 0,0,0,1,1,0,0,0,0,0,0,0,0), (2,0,1,0,0,0,0,0,0,0,0,0,0,0), (0,2,0,1,0,0,0,0,0,0,0,0,0,0)]).to(self.device)#SpeciesXReactions
        # ReactionMatLeft=np.repeat(ReactionMatLeft[None,:], self.Sites, axis=0).reshape(self.L,-1)#np.repeat(ReactionMatLeft, self.Sites,axis=0)
        ReactionMatLeft=torch.as_tensor([(1, 1,1,1,0,0), (1,0,0,0,1,0), (0,1,0,0,0,1)]).to(self.device)#SpeciesXReactions
        ReactionMatRight=torch.as_tensor([(0, 0,0,0,1,1), (2,0,1,0,0,0), (0,2,0,1,0,0)]).to(self.device)#SpeciesXReactions
        Yes=torch.as_tensor(1).to(self.device)#SpeciesXReactions
        
        ReactionMatLeftTotal_1=torch.zeros(self.L,self.Sites*6, dtype=int).to(self.device)
        ReactionMatLeftTotal_2=torch.zeros(self.L,2+(self.Sites-2)*4+2, dtype=int).to(self.device)
        ReactionMatRightTotal_1=torch.zeros(self.L,self.Sites*6, dtype=int).to(self.device)
        ReactionMatRightTotal_2=torch.zeros(self.L,2+(self.Sites-2)*4+2, dtype=int).to(self.device)
        for i in range(self.Sites):
            ReactionMatLeftTotal_1[i*int(self.L/self.Sites):(i+1)*int(self.L/self.Sites),
                                 i*6:(i+1)*6]=ReactionMatLeft    
            ReactionMatRightTotal_1[i*int(self.L/self.Sites):(i+1)*int(self.L/self.Sites),
                                 i*6:(i+1)*6]=ReactionMatRight    
    
        ReactionMatLeftTotal_2[1,0]=Yes #D_1->D_2
        ReactionMatLeftTotal_2[2,1]=Yes #L_1->L_2
        ReactionMatLeftTotal_2[self.L-2,(self.Sites-2)*4+2]=Yes #D_self.L->D_{self.L-1}
        ReactionMatLeftTotal_2[self.L-1,(self.Sites-2)*4+3]=Yes #L_self.L->L_{self.L-1}
        
        ReactionMatRightTotal_2[1+3,0]=Yes #D_1->D_2
        ReactionMatRightTotal_2[2+3,1]=Yes #L_1->L_2
        ReactionMatRightTotal_2[self.L-2-3,(self.Sites-2)*4+2]=Yes #D_self.L->D_{self.L-1}
        ReactionMatRightTotal_2[self.L-1-3,(self.Sites-2)*4+3]=Yes #L_self.L->L_{self.L-1}
        
        for i in np.arange(2,self.Sites):
            # print(i)
            ReactionMatLeftTotal_2[(i-1)*int(self.L/self.Sites)+1,(i-2)*4+2]=Yes #D_i->D_{i-1}
            ReactionMatLeftTotal_2[(i-1)*int(self.L/self.Sites)+2,(i-2)*4+3]=Yes #L_i->L_{i-1}
            ReactionMatLeftTotal_2[(i-1)*int(self.L/self.Sites)+1,(i-2)*4+4]=Yes #D_i->D_{i+1}
            ReactionMatLeftTotal_2[(i-1)*int(self.L/self.Sites)+2,(i-2)*4+5]=Yes #L_i->L_{i+1}
            
            ReactionMatRightTotal_2[(i-1)*int(self.L/self.Sites)+1+3,(i-2)*4+2]=Yes #D_i->D_{i-1}
            ReactionMatRightTotal_2[(i-1)*int(self.L/self.Sites)+2+3,(i-2)*4+3]=Yes #L_i->L_{i-1}
            ReactionMatRightTotal_2[(i-1)*int(self.L/self.Sites)+1-3,(i-2)*4+4]=Yes #D_i->D_{i+1}
            ReactionMatRightTotal_2[(i-1)*int(self.L/self.Sites)+2-3,(i-2)*4+5]=Yes #L_i->L_{i+1}

        
        ReactionMatLeftTotal=torch.cat((ReactionMatLeftTotal_1,ReactionMatLeftTotal_2),axis=1)
        ReactionMatRightTotal=torch.cat((ReactionMatRightTotal_1,ReactionMatRightTotal_2),axis=1)
        
        # print(r)
        # print(ReactionMatLeftTotal)
        # MConstrain=np.zeros(1,dtype=int)
        MConstrain=np.array([10,self.M,self.M], dtype=int).reshape(1,int(self.L/self.Sites)) #Number constrain
        # MConstrain=np.array([self.M,self.M,self.M], dtype=int).reshape(1,int(self.L/self.Sites)) #Number constrain
        MConstrain=np.repeat(MConstrain, self.Sites,axis=0).reshape(1,self.L)[0,:]
        
        #conservation=np.ones(1,dtype=int)
        conservation=initialD.sum()
        if self.binary:
            self.bits=int(np.ceil(np.log2(conservation)))
    
        IniDistri='delta'
        
        #IniDistri='poisson'
        print(conservation,MConstrain)
            
        return IniDistri,initialD,r,ReactionMatLeftTotal,ReactionMatRightTotal,MConstrain,conservation
    
