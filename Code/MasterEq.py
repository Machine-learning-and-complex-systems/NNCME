# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 08:56:03 2021

@author: James
"""
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
from Transition import DeltaFunction,TransitionState
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


plt.rc('font', size=16)

#Initialize parameters: otherwise the parameters are specified in init_out_dir-> args.py

# Parameters of the chemical system
# Following parameters are set for the example of gene expression
args.species_num=2 #Number of species in the system
args.upper_limit=100 #Upper limit of the Molecule number
args.reaction_num=4 #Number of reactions in the system
args.reaction_rates=[0.1,0.1,0.1,0.002] #Listed rates for each reaction
args.initial_distribution='delta' #Initial Distribution for the species ('delta' or 'possion' distribution is supported)
args.initial_num=[0,0] #Initial number for each species in a list (here we have zero mRNA and zero protein)
args.reaction_matrix_left=[(0, 1,1,0), (0,0,0,1)] #The reaction matrix on the left (dimension: species_num*reactions_num) (stoichiometric matrix = reaction_matrix_left-reaction_matrix_right) 
args.reaction_matrix_right=[(1, 1,0,0), (0,1,0,0)] #The reaction matrix on the right (dimension: species_num*reactions_num) (stoichiometric matrix = reaction_matrix_left-reaction_matrix_right) 
args.MConstraint=1 #Add different number constraint for each species. If the upper limit for all species is the same as args.upper_limit, then set args.Mconstraint=1
args.Conservation=1 # To maintain the total number of all species, set args.Conservation=initial_num.sum(), otherwise set it 1.

# Parameters of the variational autoregressive network (VAN)
args.batch_size=1000 #Number of samples
args.training_step=1000 #Time step of iterating the dynamical equation P_tnew=T*P_t
args.deltaT=0.1 #Time step length of iterating the dynamical equation
args.net='rnn' #Network type ('made','rnn' and 'transformer' are supported)
args.net_depth=1 #Network depth
args.net_width=16 #Network width
args.epoch1=3000 #Maximum number of steps first time step (usually larger to ensure the accuracy)
args.epoch2=100 #Maximum number of steps of later time steps
args.saving_data_time_step=[0,1e2,5e2,2e3,1e4,2e4,5e4,1e5,1.5e5,2e5,2.5e5,3e5,3.5e5,4e5,5e5,6e5,7e5,8e5,9e5,1e6] #To save data at which time steps (give in a list)
args.training_loss_print_step=[0,1,2,101,1001,2e3,1e4,1e5,2e5,3e5,4e5,5e5] #To print training loss at which time steps (give in a list)


class ModelName:
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
        
        self.L=args.species_num
        initialD=np.array(args.initial_num).reshape(1,self.L)#0.1#0.1 # the parameter for the initial Poisson distribution
        IniDistri=self.IniDistri=args.initial_distribution
        print(self.IniDistri)
        
        r=torch.zeros(args.reaction_num) #Reaction rates
        for i in range(args.reaction_num):r[i]=args.reaction_rates[i]

        ReactionMatLeft=torch.as_tensor(args.reaction_matrix_left).to(self.device)#SpeciesXReactions
        ReactionMatRight=torch.as_tensor(args.reaction_matrix_right).to(self.device)#SpeciesXReactions
        
        if args.MConstraint==1:MConstrain=np.zeros(args.MConstraint,dtype=int)
        else:MConstrain=np.array(args.MConstraint,dtype=int)
        conservation=np.ones(args.Conservation,dtype=int)
        # Stoichiometric matrix
        #V=ReactionMatRight-ReactionMatLeft #SpeciesXReactions    
        return IniDistri,initialD,r,ReactionMatLeft,ReactionMatRight,MConstrain,conservation


def Optimizer(net,args):

    params = list(net.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = int(sum([np.prod(p.shape) for p in params]))
    my_log('Total number of trainable parameters: {}'.format(nparams))
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))

    return optimizer,params,nparams
         
          
def Test():  
    # # # # # # #Initialize parameters: otherwise the parameters are specified in init_out_dir-> args.py
    args.L=args.species_num
    args.M=int(args.upper_limit)#Upper limit of the Molecule number
    
    args.batch_size=args.batch_size
    args.Tstep=args.training_step#Time step of iterating the dynamical equation P_tnew=T*P_t, where T=(I+W*delta t)
    args.delta_t=args.deltaT    
    args.net =args.net
    args.net_depth=args.net_depth# including output layer and not input data layer
    args.net_width=args.net_width
    args.max_stepAll=args.epoch1
    args.max_stepLater=args.epoch2

    args.binary=False
    args.AdaptiveT=False
    args.AdaptiveTFold=5
    args.print_step=20
    args.saving_data_time_step=args.saving_data_time_step
    args.training_loss_print_step=args.training_loss_print_step

    #Default parameters:
    args.bias=True
    args.bits=1
    if args.binary:
        args.bits=int(np.ceil(np.log2(args.M)))
    args.Percent=0.2         
    args.clip_grad=1
    args.dtype='float64'
    args.epsilon=1e-300#initial probability of zero species number
    args.lr_schedule=False#True
    start_time2 = time.time()


    model = ModelName(**vars(args))   
    
    args.IniDistri,args.initialD,r,ReactionMatLeft,ReactionMatRight,args.MConstrain,args.conservation=model.rates()
    print(args.initialD)
    print(args.conservation)
    print(args.MConstrain)
    if args.Model=='homo1' or args.Model=='ToggleSwitch':
       args.epsilon=1e-30

      
    cc=torch.as_tensor(r, dtype=default_dtype_torch).to(args.device)
    V=ReactionMatRight-ReactionMatLeft #SpeciesXReactions    
    
    args.size=args.L # the number of spin: 1D, doesnt' count the boundary spins
    for delta_tt in np.arange(1): #The result is not sensitively depend on time steplength so far
        start_time = time.time()
        init_out_dir()
        if args.clear_checkpoint:
            clear_checkpoint()
        last_step = get_last_checkpoint_step()
        print(last_step)
        if last_step >= 0:
            my_log('\nCheckpoint found: {}\n'.format(last_step))
        else:
            clear_log()    
        
        SummaryListDynPartiFuncLog2=[]
        SummaryListDynPartiFuncLog3=[]
        SummaryLoss1=[]  
        SummaryLoss2=[]
        net_new = []
        
        for lambda_tilt in np.arange(1):
            args.lambda_tilt=lambda_tilt
            args.max_step=args.max_stepAll
            #Initialize net and optimizer  
            if args.net == 'rnn':
                net = GRU(**vars(args))
            elif args.net == 'transformer':
                net = TraDE(**vars(args)).to(args.device)    
            else:
                raise ValueError('Unknown net: {}'.format(args.net))
            net.to(args.device)
            

            ListDynPartiFuncLog3=[]
            Loss1=[] 
            SampleSum=[]
            Loss2=[]
            delta_TSum=[]
            
            #Load NN starting from certain time point:
            startT=0
            if args.loadVAN:
                PATH=args.out_filename+'Tstep'+str(args.loadTime)
                startT=args.loadTime
                print(PATH)
                if args.cuda==-1:
                    state = torch.load(PATH,map_location=torch.device('cpu') ) #CPU  
                else:
                    state = torch.load(PATH) #GPU
                net1=GRU(**vars(args))
                net1.load_state_dict(state['net']) #new save format 
                net1.to(args.device)
                net_new=copy.deepcopy(net1)#net
                net_new.to(args.device)
                net_new.requires_grad=False
                print('Use saved VAN')          
                #net_new=state
                net=copy.deepcopy(net1)
                args.out_filename=args.out_filename + 'loadVAN'+str(args.loadTime)#+ 'lr'+str(args.lrLoad)+ 'epoch'+str(args.max_stepLoad)
                ensure_dir(args.out_filename+ '_img/')
                
            
            optimizer,params,nparams = Optimizer(net,args)
            
            
            for Tstep in range(startT,args.Tstep): # Time step of the dynamical equation
                ### Load parameters of last training: if necessary
                if args.lr_schedule:  
                    if args.lr_schedule_type==1:
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, factor=0.5, patience=int(args.max_step*args.Percent), verbose=True, threshold=1e-4, min_lr=1e-5)   
                    if args.lr_schedule_type==2:
                        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch*10*args.lr+1))
                    
                # Start
                init_time = time.time() - start_time
                sample_time = 0
                train_time = 0
                start_time = time.time()       
                
                ListDistanceCheck_Eucli=[]
                ListDistanceCheck=[]
                Listloss_mean=[]
                Listloss_std=[]
                if Tstep>=1:
                    args.max_step=args.max_stepLater
                    if args.loadVAN and Tstep==startT:
                        args.max_step=args.max_stepAll
                
                #Train VAN
                SampleT=[]
                for step in range(last_step + 1, args.max_step + 1):    
                    optimizer.zero_grad()   
                    sample_start_time = time.time()
                    with torch.no_grad():
                        sample, x_hat = net.sample(args.batch_size)

                    sample_time += time.time() - sample_start_time
                    train_start_time = time.time()
                    
                    log_prob = net.log_prob(sample) #sample has size batchsize X 1 X systemSize                
                    if step==0:
                        delta_T=1
                    with torch.no_grad():
                        LogTP_t,delta_T=TransitionState(sample,args,Tstep,step,net_new,ReactionMatLeft,V,cc,delta_T,model)#.detach()
                        TP_t_normalize=(torch.exp(LogTP_t)/(torch.exp(LogTP_t)).sum()*(torch.exp(log_prob)).sum()).detach()
                        loss= log_prob-LogTP_t.detach()             
                        lossL2 = (torch.exp(log_prob)-TP_t_normalize)
                        losshe = -torch.sqrt(torch.exp(log_prob)*TP_t_normalize)
                    
                    assert not LogTP_t.requires_grad
                    if args.lossType=='kl':
                        loss_reinforce = torch.mean((loss - loss.mean())*log_prob)                      
                    elif args.lossType=='klreweight':
                        loss3=torch.exp(log_prob)*(loss)/torch.exp(log_prob).mean()
                        loss_reinforce = torch.mean((loss3 - loss3.mean())*log_prob)                      
                    elif args.lossType=='l2':
                        loss_reinforce = torch.mean((lossL2 - lossL2.mean())*log_prob)        
                    elif args.lossType=='he':
                        loss_reinforce = torch.mean((losshe)*log_prob)
                    elif args.lossType=='ss':
                        loss_reinforce = torch.mean((loss - loss.mean())*log_prob/2) # steady state# Conversion from probability P to state \psi           
                    loss_reinforce.backward() 
                    
                
                    if args.clip_grad:
                        nn.utils.clip_grad_norm_(params, args.clip_grad)           
                    optimizer.step()          
                    if args.lr_schedule:
                        scheduler.step(loss.mean())     
                    train_time += time.time() - train_start_time
                    
                    loss_std = loss.std()
                    loss_mean=loss.mean()
                    Listloss_mean.append(loss_mean.detach().cpu().numpy())
                    Listloss_std.append(loss_std.detach().cpu().numpy())
                    DistanceCheck_Eucli=torch.sqrt(torch.sum((torch.exp(net.log_prob(sample))-TP_t_normalize)**2))
                    DistanceCheck=torch.nn.functional.kl_div(net.log_prob(sample),TP_t_normalize, None, None, 'sum')#function kl_div is not the same as wiki's explanation. 
                    ListDistanceCheck_Eucli.append(DistanceCheck_Eucli.detach().cpu().numpy())
                    ListDistanceCheck.append(DistanceCheck.detach().cpu().numpy())
                    if step>int(args.max_step-10):
                        with torch.no_grad():
                            SampleT.append(np.array(sample.detach().cpu()))
                            
                    #print out:
                    if args.print_step and step % args.print_step == 0 and Tstep % int(args.print_step) == 0:
                        if step > 0:
                            sample_time /= args.print_step
                            train_time /= args.print_step
                        used_time = time.time() - start_time
                        sample_time = 0
                        train_time = 0
                        
                with torch.no_grad():
                    if Tstep % int(args.print_step)==0: #if Tstep % (1/args.delta_t)==0: 
                        Loss1.append(np.array(loss_mean.detach().cpu()))      
                        Loss2.append(np.array(DistanceCheck_Eucli.detach().cpu()))    
                        SampleSum.append(np.array(SampleT).reshape(-1,args.L))
                    delta_TSum.append(delta_T)
                    net_new=copy.deepcopy(net)#net
                    net_new.requires_grad=False
                    if Tstep in args.saving_data_time_step:#[0,1e2,5e2,2e3,1e4,2e4,5e4,1e5,1.5e5,2e5,2.5e5,3e5,3.5e5,4e5,5e5,6e5,7e5,8e5,9e5,1e6]:
                    # if (Tstep==0 or Tstep==100 or Tstep==500 or Tstep==2000 or Tstep==10000 or Tstep==20000 or Tstep==50000 or Tstep==100000 or Tstep==150000 or Tstep==200000 or Tstep==250000 or Tstep==300000 or Tstep==350000 or Tstep==400000 or Tstep==500000 or Tstep==600000 or Tstep==700000 or Tstep==800000 or Tstep==900000 or Tstep==1000000):# and not args.loadVAN:
                        PATH=args.out_filename+'Tstep'+str(Tstep)
                        #torch.save(net, PATH)
                        state = {'net': net.state_dict(),
                                'optimizer': optimizer.state_dict()}
                        torch.save(state, PATH)
                        argsSave=[args.Tstep,args.delta_t,args.L,args.M,args.lr,args.initialD,r.detach().cpu().numpy(),args.print_step]
                        SummaryListDynPartiFuncLog2=np.array(SummaryListDynPartiFuncLog2)
                        SummaryLoss1=[]
                        SummaryLoss2=[]
                        np.savez('{}_img/Data'.format(args.out_filename),SummaryListDynPartiFuncLog2,argsSave,SummaryListDynPartiFuncLog3,SummaryLoss1,SummaryLoss2,np.array(SampleSum),np.array(delta_TSum))     
                    
                    
                        
                #Plot training loss
                if Tstep in args.training_loss_print_step:#[0,1,2,101,1001,2e3,1e4,1e5,2e5,3e5,4e5,5e5]:
                # if Tstep<3 or Tstep==101 or Tstep==1001 or Tstep==2000 or Tstep==10000 or Tstep==100000 or Tstep==200000 or Tstep==300000 or Tstep==400000 or Tstep==500000:
                    plt.figure(num=None,  dpi=300, edgecolor='k')
                    fig, axes = plt.subplots(2,1)
                    fig.tight_layout()
                    ax = plt.subplot(2,1, 1)
                    plt.plot(range(last_step + 1, args.max_step + 1), Listloss_mean, label='Loss_mean')
                    plt.plot(range(last_step + 1, args.max_step + 1), Listloss_std, label='Loss_std')
                    axes = plt.gca()
                    axes.set_yscale('log')
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')
                    plt.legend()        
                    ax = plt.subplot(2,1, 2)
                    axes = plt.gca()
                    axes.set_yscale('log')
                    plt.plot(range(last_step + 1, args.max_step + 1), ListDistanceCheck_Eucli, label='Euclidean distance')
                    plt.plot(range(last_step + 1, args.max_step + 1), ListDistanceCheck, label='KL-divergence')
                    plt.ylabel('Distance')
                    plt.xlabel('Epoch')
                    fig = plt.gcf()
                    plt.legend()
                    fig.set_size_inches(8, 8)
                    plt.tight_layout()        
                    plt.savefig('{}_img/TimeStep{}.jpg'.format(args.out_filename, Tstep), dpi=300)

            SummaryLoss1.append(Loss1)      
            SummaryLoss2.append(Loss2) 
            SummaryListDynPartiFuncLog3.append(ListDynPartiFuncLog3)
            np.savez('{}_img/Data'.format(args.out_filename),np.array(SummaryListDynPartiFuncLog2),np.array(SampleSum),np.array(delta_TSum))    
    
    print(np.array(SampleSum).shape)
    end_time2 = time.time()  
    print('Time ',(end_time2-start_time2)/60)
    print('Time ',(end_time2-start_time2)/3600)
    TimeEnd=np.array((end_time2-start_time2)/3600)
    argsSave=[args.Tstep,args.delta_t,args.L,args.M,args.lr,args.initialD,r.detach().cpu().numpy(),args.print_step,TimeEnd]
      
    SummaryListDynPartiFuncLog2=np.array(SummaryListDynPartiFuncLog2)
    SummaryLoss1=np.array(SummaryLoss1)
    SummaryLoss2=np.array(SummaryLoss2)
    print(np.array(delta_TSum))
    np.savez('{}_img/Data'.format(args.out_filename),SummaryListDynPartiFuncLog2,argsSave,SummaryListDynPartiFuncLog3,SummaryLoss1,SummaryLoss2,np.array(SampleSum),np.array(delta_TSum))     
    
if __name__ == '__main__':
    Test()    
    
    
    
    
    
