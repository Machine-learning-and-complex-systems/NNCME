from scipy.stats import poisson,zipf
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import copy
from args import args
#from made1D import MADE1D
from gru import GRU
from transformer import TraDE
from utils import (
    clear_checkpoint,
    clear_log,
    get_last_checkpoint_step,
    ignore_param,
    init_out_dir,
    my_log,
    print_args,
    ensure_dir,
)
from utils import default_dtype_torch

def DeltaFunction(args,X,Y):
    factor=0.01 #not too large to make a delta distribution, not too small to avoid numerical instablity of the perfect delta distribution
    Prob=torch.zeros_like(X, dtype=default_dtype_torch).to(args.device)
    #Given non-delta's number a small probability
    #print((factor/(args.MConstrain-1)).reshape(1,-1))
    if args.MConstrain[0]==0:
        args.MConstrain=args.M*np.ones(args.L, dtype=int)#args.MConstrain->args.L
    if len(X.shape)==2:
        SmallProb=torch.tensor((factor/(args.MConstrain-1)).reshape(1,-1).repeat(args.batch_size, axis=0), dtype=default_dtype_torch).to(args.device)  
    if len(X.shape)==3:
        SmallProb=torch.tensor((factor/(args.MConstrain-1)).reshape(1,-1,1).repeat(args.batch_size, axis=0).repeat(X.shape[2], axis=2), dtype=default_dtype_torch).to(args.device)  
    Prob[X-Y==0]=torch.tensor(1-factor, dtype=default_dtype_torch).to(args.device)
    # print(Prob.shape,SmallProb.shape)
    Prob[X-Y!=0]=SmallProb[X-Y!=0]
    return Prob
    
    
def TransitionState(sample,args,Tstep,step,net_new,ReactionMatLeft,V,cc,delta_T,model):
    Sample1D=(sample.view(-1, args.size))  #sample has size batchsize X systemSize
    #All possible configurations reacted to the sampled state: BatchSize X SystemSize  X Reactions
    SampleNeighbor1D=Sample1D.repeat(V.shape[1],1,1).permute(1,2,0)
    SampleNeighbor1D_Win=SampleNeighbor1D-V # States that transit into the sampled states by each of the reation
    UpBoundary=args.M
    if args.conservation>1 and args.Sites==1:
        UpBoundary=args.conservation-1 
    if args.MConstrain[0]>0:
        UpBoundary=torch.tensor(args.MConstrain, dtype=SampleNeighbor1D_Win.dtype).to(args.device) 
        UpBoundary=UpBoundary.view(-1,1).repeat(SampleNeighbor1D.shape[0],1,SampleNeighbor1D.shape[2]) 
        SampleNeighbor1D_Win[SampleNeighbor1D_Win>=UpBoundary]=UpBoundary[SampleNeighbor1D_Win>=UpBoundary]#-1
    NotHappen_in_low=SampleNeighbor1D_Win < 0
    SampleNeighbor1D_Win[NotHappen_in_low] = 0 # Make the negative number of species to zero
    NotHappen_in_up=SampleNeighbor1D_Win >= UpBoundary  
    SampleNeighbor1D_Wout=SampleNeighbor1D+V# States that transit out from the sampled states by each of the reation
    if args.MConstrain[0]>0:
        SampleNeighbor1D_Win[NotHappen_in_up] = UpBoundary[NotHappen_in_up]-1
        SampleNeighbor1D_Wout[SampleNeighbor1D_Wout>=UpBoundary]=UpBoundary[SampleNeighbor1D_Wout>=UpBoundary]#-1
    else:
        SampleNeighbor1D_Win[NotHappen_in_up] =UpBoundary-1
    NotHappen_out_low=SampleNeighbor1D_Wout < 0
    NotHappen_out_up=SampleNeighbor1D_Wout >= UpBoundary
    
    #Propensity: Make product of chemical flux and reaction rates
    Win=SampleNeighbor1D_Win**ReactionMatLeft # Reaction in-flux by [Species]^Stoichiometric
    Win[NotHappen_in_low]=0 # Make the not-happen reaction with zero chemical species, giving zero flux when multiplied below
    Win[NotHappen_in_up]=0 # Make the not-happen reaction with zero chemical species, giving zero flux when multiplied below
    #Check right boundary: done, reflecting boundary condition   
    Wout=SampleNeighbor1D**ReactionMatLeft# Reaction out-flux by [Species]^Stoichiometric
    Wout[NotHappen_out_low]=0 # Make the not-happen reaction with zero chemical species, giving zero flux when multiplied below
    Wout[NotHappen_out_up]=0 # Make the not-happen reaction with zero chemical species, giving zero flux when multiplied below
    
    Propensity_in,Propensity_out=model.Propensity(Win,Wout,cc,SampleNeighbor1D_Win,SampleNeighbor1D,NotHappen_in_low,NotHappen_in_up,NotHappen_out_low,NotHappen_out_up)
    # Propensity_in=torch.prod(Win,1)*cc#torch.tensor(r, dtype=default_dtype_torch).to(args.device)   
    # Propensity_out=torch.prod(Wout,1)*cc#torch.tensor(r, dtype=default_dtype_torch).to(args.device)    
    R=torch.sum(Propensity_out,1)
    
    if Tstep==0: #initial distribution: depending on the system
        with torch.no_grad():
            if args.IniDistri=='delta':
                # #Manually generate delta-distribution
                LogP_t=torch.sum(torch.log(torch.as_tensor(DeltaFunction(args,Sample1D.cpu(),args.initialD.repeat(args.batch_size, axis=0)),dtype=default_dtype_torch)+args.epsilon).to(args.device),1)#[:,0]#torch.tensor(np.exp(-args.initialD)*args.initialD**Sample1D/math.factorial(int(Sample1D)),dtype=default_dtype_torch).to(args.device)
                LogP_t_other=torch.sum(torch.log(torch.as_tensor(DeltaFunction(args,SampleNeighbor1D_Win.cpu(),args.initialD.repeat(args.batch_size, axis=0).reshape(args.batch_size,args.L,1).repeat(ReactionMatLeft.shape[1], axis=2)),dtype=default_dtype_torch)+args.epsilon).to(args.device),1)#[:,0,:]#torch.tensor(np.exp(-args.initialD)*args.initialD**Sample1D/math.factorial(int(Sample1D)),dtype=default_dtype_torch).to(args.device)
            elif args.IniDistri=='zipf':
                # #Zipf distribution to generate delta-distribution
                LogP_t=torch.sum(torch.log(torch.as_tensor(zipf.pmf(Sample1D.cpu()-args.initialD.repeat(args.batch_size, axis=0),6),dtype=default_dtype_torch)+args.epsilon).to(args.device),1)#[:,0]#torch.tensor(np.exp(-args.initialD)*args.initialD**Sample1D/math.factorial(int(Sample1D)),dtype=default_dtype_torch).to(args.device)
                LogP_t_other=torch.sum(torch.log(torch.as_tensor(zipf.pmf(SampleNeighbor1D_Win.cpu()-args.initialD.repeat(args.batch_size, axis=0).reshape(args.batch_size,args.L,1).repeat(ReactionMatLeft.shape[1], axis=2),6),dtype=default_dtype_torch)+args.epsilon).to(args.device),1)#[:,0,:]#torch.tensor(np.exp(-args.initialD)*args.initialD**Sample1D/math.factorial(int(Sample1D)),dtype=default_dtype_torch).to(args.device)        
            elif args.IniDistri=='MM':
                Temp11=poisson.pmf(k=Sample1D[:,:2].cpu(), mu=args.initialD)#substrate,#enzyme
                Temp13=poisson.pmf(k=Sample1D[:,2:].cpu(), mu=0.1) #a complex, a product,
                Temp1=np.concatenate((Temp11,Temp13),axis=1)
                LogP_t=torch.sum(torch.log(torch.as_tensor(Temp1,dtype=default_dtype_torch).to(args.device)),1)#[:,0]#torch.tensor(np.exp(-args.initialD)*args.initialD**Sample1D/math.factorial(int(Sample1D)),dtype=default_dtype_torch).to(args.device)
                Temp21=poisson.pmf(k=SampleNeighbor1D_Win[:,:2,:].cpu(), mu=args.initialD)
                Temp22=poisson.pmf(k=SampleNeighbor1D_Win[:,2:,:].cpu(), mu=0.1)
                Temp2=np.concatenate((Temp21,Temp22),axis=1)
                LogP_t_other=torch.sum(torch.log(torch.as_tensor(Temp2,dtype=default_dtype_torch).to(args.device)),1)#[:,0,:]#torch.tensor(np.exp(-args.initialD)*args.initialD**Sample1D/math.factorial(int(Sample1D)),dtype=default_dtype_torch).to(args.device)
            else:
                if args.L==1:
                    initialD2=args.initialD.repeat(args.batch_size, axis=0).reshape(-1,1)
                else:
                    initialD2=args.initialD.repeat(args.batch_size, axis=0)
                LogP_t=torch.sum(torch.log(torch.as_tensor(poisson.pmf(k=Sample1D.cpu(), mu=initialD2),dtype=default_dtype_torch)+args.epsilon).to(args.device),1)#[:,0]#torch.tensor(np.exp(-args.initialD)*args.initialD**Sample1D/math.factorial(int(Sample1D)),dtype=default_dtype_torch).to(args.device)
                LogP_t_other=torch.sum(torch.log(torch.as_tensor(poisson.pmf(k=SampleNeighbor1D_Win.cpu(), mu=args.initialD.repeat(args.batch_size, axis=0).reshape(args.batch_size,args.L,1).repeat(ReactionMatLeft.shape[1], axis=2)),dtype=default_dtype_torch)+args.epsilon).to(args.device),1)#[:,0,:]#torch.tensor(np.exp(-args.initialD)*args.initialD**Sample1D/math.factorial(int(Sample1D)),dtype=default_dtype_torch).to(args.device)
 
    if Tstep>0:
        with torch.no_grad():
            LogP_t=net_new.log_prob(sample).detach()
            Temp=torch.transpose(SampleNeighbor1D_Win, 1, 2)#.view(sample.shape[0], args.size, args.size) #BatchSize X NeighborSize X SystemSize      
            if args.net == 'rnn' or args.net == 'transformer':
                Temp3=torch.reshape(Temp, (args.batch_size*cc.shape[0],args.size)) #For RNN
            else:
                Temp3=torch.reshape(Temp, (args.batch_size*cc.shape[0],1,args.size)) #For RNN
            LogP_t_otherTemp=net_new.log_prob(Temp3).detach()
            LogP_t_other=torch.reshape(LogP_t_otherTemp, (args.batch_size,cc.shape[0]))
            
    if args.cuda==-1:
        if step%50==0:# For my computer
            print(torch.mean(Sample1D,0),torch.mean(torch.mean(Sample1D,0)),Tstep,step)
    else: 
        if step==40 and Tstep%10==0:#For server
            print(torch.mean(Sample1D,0),torch.mean(torch.mean(Sample1D,0)),Tstep,step)

    with torch.no_grad():
        if args.AdaptiveT: #only at the first 10 epoch, and record it
            if step==0:
                delta_T=args.delta_t*args.AdaptiveTFold
            Temp2=1+(torch.sum(torch.exp(LogP_t_other-LogP_t.repeat(cc.shape[0],1).t())*Propensity_in,1)-R)*delta_T
            if step<=10:
                while torch.min(Temp2)<0:
                    delta_T=delta_T/2
                    Temp2=1+(torch.sum(torch.exp(LogP_t_other-LogP_t.repeat(cc.shape[0],1).t())*Propensity_in,1)-R)*delta_T
                    if delta_T<=args.delta_t:
                        print('reduce delta t at ',step)
                        Temp2[Temp2<=0]=args.epsilon
                        break
            else:
                Temp2=1+(torch.sum(torch.exp(LogP_t_other-LogP_t.repeat(cc.shape[0],1).t())*Propensity_in,1)-R)*delta_T
                Temp2[Temp2<=0]=args.epsilon
        else:   
            if Tstep==0:
                Temp2=1+(torch.sum(torch.exp(LogP_t_other-LogP_t.repeat(cc.shape[0],1).t())*Propensity_in,1)-R)*args.delta_t/10#/4 #
                # if step==0:
                #     print('First Tstep to have smaller delta_t to reduce loss.mean')
            else:
                Temp2=1+(torch.sum(torch.exp(LogP_t_other-LogP_t.repeat(cc.shape[0],1).t())*Propensity_in,1)-R)*args.delta_t
            
            if torch.min(Temp2)<0:
                print('reduce delta t at ',step)
                Temp2[Temp2<=0]=args.epsilon

        LogTP_t1=torch.log(Temp2)+LogP_t    

    return LogTP_t1,delta_T#TP_t