"""System definition for ffl."""

import numpy as np
import torch

class FFL:

    '''Chemical system class for ffl.'''
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
        Propensity_in=torch.prod(Win,1)*cc#torch.tensor(r, dtype=torch.float64).to(self.device)   
        Propensity_out=torch.prod(Wout,1)*cc#torch.tensor(r, dtype=torch.float64).to(self.device)    
    
        return Propensity_in,Propensity_out
    
    
    def rates(self): 
        self.L=9 

        # MConstrain=np.zeros(1,dtype=int)
        MConstrain=np.array([2, 2, 2, self.M,self.M,self.M, 2, 2, 2], dtype=int) #Number constrain
        conservation=np.ones(1,dtype=int)
        
        sA=sB=sC=10
        dA=dB=dC=1
        rbA=rcA=rcB=0.005
        fbA=fcA=fcB=0.1
        
        if self.Para==1:
        # bimodality in both proteins B and C
            k1=3.0
            k2=0.5
            k3=5.0
        if self.Para==2:
        #  tri-modality in protein C and bimodality in protein B
            k1=0.1
            k2=2.75
            k3=5.0



        r=torch.tensor([sA, dA, sB, dB, sC, dC, rbA, fbA, sB*k1, rcA, fcA, sC*k3, rcB, fcB, sC*k2])
        print(r.shape)
        initialD=np.zeros(shape=(1,self.L),dtype=int) #for zipf to realize delta
        initialD[0,0]=1
        initialD[0,1]=1
        initialD[0,2]=1
        print(initialD)
        # Reaction matrix #SpeciesXReactions


        ReactionMatLeft=torch.as_tensor([
        (1, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 1, 0, 0, 0),
        (0, 1, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 1, 0, 0),
        (0, 0, 1, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 1, 0),
        (0, 0, 0, 0, 0, 0, 0, 1, 0),
        (0, 0, 1, 0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 0, 1),
        ]).to(self.device).t()
        
        ReactionMatRight=torch.as_tensor([
        (1, 0, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 1, 0, 0),
        (0, 1, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 1, 0),
        (0, 0, 1, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 1, 0, 1, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 1, 0, 1, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 1, 0, 0, 1),
        ]).to(self.device).t()
        
        IniDistri='delta'
        conservation=np.ones(1,dtype=int)
        
        return IniDistri,initialD,r,ReactionMatLeft,ReactionMatRight,MConstrain,conservation
    
