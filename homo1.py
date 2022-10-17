# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 17:35:58 2022

@author: james
"""

import numpy as np
import torch

class homo1:
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
        
    def Propensity(self,Win,Wout,cc,SampleNeighbor1D_Win,SampleNeighbor1D,NotHappen_in_low,NotHappen_in_up,NotHappen_out_low,NotHappen_out_up):
        Propensity_in=torch.prod(Win,1)*cc#torch.tensor(r, dtype=torch.float64).to(args.device)   
        Propensity_out=torch.prod(Wout,1)*cc#torch.tensor(r, dtype=torch.float64).to(args.device)    
    
        return Propensity_in,Propensity_out
    
        
    
    
    def rates(self): 
         
        self.L=3#10#10#16 # Lattice size: 1D  
        r=torch.zeros(6) #Reaction rates   
        # Parameter values
        ka = 1
        kn = 1
        kd = 1
        #self.Para=0.1
        V = self.Para#0.1#1#10
        # D=int(self.M/3-1)
        D=int((self.M-2)/2)
        r[0] = ka/V #kr
        r[1] = ka/V #kp
        r[2] = kn #yr
        r[3] = kn #yp  
        r[4] = kd #yr
        r[5] = kd #yp  
        #self.initialD=np.array([self.M-2*D,D,D]).reshape(1,self.L)#0.1#0.1 # the parameter for the initial Poisson distribution
        #self.initialD=np.array([self.M-2*D,D+1,D-1]).reshape(1,self.L)#0.1#0.1 # the parameter for the initial Poisson distribution
        #self.initialD=np.array([D,D+1,D-1]).reshape(1,self.L)#0.1#0.1 # the parameter for the initial Poisson distribution
        #self.initialD=np.array([int(20),D-int(10),D-int(20)]).reshape(1,self.L)#0.1#0.1 # the parameter for the initial Poisson distribution
        #self.initialD=np.array([D,D+int(D/2),D-int(D/2)]).reshape(1,self.L)#0.1#0.1 # the parameter for the initial Poisson distribution
        #self.initialD=np.array([1,D+int(D/2),D-int(D/2)]).reshape(1,self.L)#0.1#0.1 # the parameter for the initial Poisson distribution
        #self.initialD=np.array([1,D+1,D-1]).reshape(1,self.L)#0.1#0.1 # the parameter for the initial Poisson distribution
        #initialD=np.array([D,D+1,D-1]).reshape(1,self.L)#0.1#0.1 # the parameter for the initial Poisson distribution
        #initialD=np.array([D,D,D]).reshape(1,self.L)#0.1#0.1 # the parameter for the initial Poisson distribution
        initialD=np.array([2,D,D]).reshape(1,self.L)#0.1#0.1 # the parameter for the initial Poisson distribution
        # Reaction matrix
        ReactionMatLeft=torch.as_tensor([(1, 1,1,1,0,0), (1,0,0,0,1,0), (0,1,0,0,0,1)]).to(self.device)#SpeciesXReactions
        ReactionMatRight=torch.as_tensor([(0, 0,0,0,1,1), (2,0,1,0,0,0), (0,2,0,1,0,0)]).to(self.device)#SpeciesXReactions
        # MConstrain=np.zeros(1,dtype=int)
        MConstrain=np.array([10,self.M,self.M], dtype=int) #Number constrain
        
        #conservation=np.ones(1,dtype=int)
        conservation=initialD.sum()
        if self.binary:
            self.bits=int(np.ceil(np.log2(conservation)))
    
        IniDistri='delta'
        
        #IniDistri='poisson'
        print(conservation,self.bits)
            
        return IniDistri,initialD,r,ReactionMatLeft,ReactionMatRight,MConstrain,conservation
    
