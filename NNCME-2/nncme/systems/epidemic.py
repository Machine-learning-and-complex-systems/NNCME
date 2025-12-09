"""System definition for epidemic."""

import torch
import numpy as np
import math

class Epidemic:
    

    '''Chemical system class for epidemic.'''
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
        Propensity_in=torch.prod(Win,1)*cc#torch.tensor(r, dtype=torch.float64).to(args.device)   
        Propensity_out=torch.prod(Wout,1)*cc#torch.tensor(r, dtype=torch.float64).to(args.device) 
        
        return Propensity_in,Propensity_out
        
    def rates(self,Tstep,delta_t): 

        IniDistri='delta'
        r=np.zeros(6)
        r[0] = 0.003
        r[1] = 0.02
        r[2] = 0.007
        r[3] = 0.002
        r[4] = 0.05
        r[5] = 0.002 
        
        c0=0.003
        epsilon=0.2#0,0.2,0.6
        T=6
        omega=2*math.pi/T
        r[0] = c0*(1+epsilon*math.sin(omega*Tstep*delta_t))
        
        initialD=np.array([50,10,0]).reshape(1,self.L)#0.1#0.1 # the parameter for the initial Poisson distribution
        # Reaction matrix
        ReactionMatLeft=torch.as_tensor([(1,0,0,1,0,0),(1,1,0,0,1,0),(0,0,1,0,0,1)]).to(self.device)#SpeciesXReactions
        ReactionMatRight=torch.as_tensor([(0,0,1,0,0,0),(2,0,0,0,0,0),(0,1,0,0,0,0)]).to(self.device)#SpeciesXReactions
        
        MConstrain=np.zeros(1,dtype=int)
        conservation=np.ones(1,dtype=int)#initialD.sum()#np.ones(1,dtype=int)
        
        # Stoichiometric matrix
        #V=ReactionMatRight-ReactionMatLeft #SpeciesXReactions    
        return IniDistri,initialD,r,ReactionMatLeft,ReactionMatRight,MConstrain,conservation
