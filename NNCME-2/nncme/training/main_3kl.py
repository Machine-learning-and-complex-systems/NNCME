"""KL objective training loop comparing different KL variants."""

import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from scipy.stats import poisson, zipf
from torch import nn

from nncme.args import args
from nncme.networks.gru import GRU
from nncme.networks.transformer import TraDE
from nncme.training.transition_three_kls import TransitionState3, TransitionState4
from nncme.utils import (
    clear_checkpoint,
    clear_log,
    get_last_checkpoint_step,
    ignore_param,
    init_out_dir,
    my_log,
    print_args,
    ensure_dir,
)
from nncme.utils import default_dtype_torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
plt.rc("font", size=16)
def DeltaFunction(args,X,Y):
    """Deltafunction operation.

    

    Args:

        args: Parameter forwarded to DeltaFunction.

        X: Parameter forwarded to DeltaFunction.

        Y: Parameter forwarded to DeltaFunction.

    

    Returns:

        Result produced by DeltaFunction.

    """


    factor=args.delta_factor #not too large to make a delta distribution, not too small to avoid numerical instablity of the perfect delta distribution
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
    Prob[X-Y!=0]=SmallProb[X-Y!=0]
    return Prob

def Optimizer(net,args):


    params = list(net.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = int(sum([np.prod(p.shape) for p in params]))
    my_log('Total number of trainable parameters: {}'.format(nparams))
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))

    return optimizer,params,nparams

                   
def Test(model):      

    start_time2 = time.time()    
        
    if args.Model!='Epidemic':
        args.IniDistri,args.initialD,r,ReactionMatLeft,ReactionMatRight,args.MConstrain,args.conservation=model.rates()
        cc=torch.as_tensor(r, dtype=default_dtype_torch).to(args.device)
        V=ReactionMatRight-ReactionMatLeft #SpeciesXReactions 
    print('args.initialD:',args.initialD)
    print('args.conservation:',args.conservation)
    print('args.MConstrain:',args.MConstrain)
       
     
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
            Lossmean=[]
            Lossstd=[]
            
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
                
            net_new=copy.deepcopy(net)#net
            net_new.to(args.device)
            net_new.requires_grad=False
            optimizer,params,nparams = Optimizer(net,args)
            
            
            for Tstep in range(startT,args.Tstep): # Time step of the dynamical equation
                if args.Model=='Epidemic':
                    args.IniDistri,args.initialD,r,ReactionMatLeft,ReactionMatRight,args.MConstrain,args.conservation=model.rates(Tstep,args.delta_t)
                    cc=torch.as_tensor(r, dtype=default_dtype_torch).to(args.device)
                    V=ReactionMatRight-ReactionMatLeft #SpeciesXReactions    
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
                
                Listloss_mean=[]
                Listloss_std=[]
                Listloss_forwardKL=[]
                Listloss_reverseKL=[]
                if Tstep>=1:
                    args.max_step=args.max_stepLater
                    if args.loadVAN and Tstep==startT:
                        args.max_step=args.max_stepAll
                
                #Train VAN
                SampleT=[]
                delta_T=1

                
                for step in range(last_step + 1, args.max_step + 1):    
                    
                    optimizer.zero_grad()   
                    sample_start_time = time.time()
                        
                    sample_time += time.time() - sample_start_time
                    train_start_time = time.time()
                    
                    if step==0 or step==33 or step==66:
                        with torch.no_grad():
                            sample, x_hat = net_new.sample(args.batch_size) # sample from P_t
                            if args.lossType=='ForwardKL3' or args.lossType=='ReverseKL2':sample2=sample
                            else:sample2=TransitionState3(sample,args,Tstep,net_new,ReactionMatLeft,V,cc,delta_T,model)#sample from TP_t
                            log_prob0 = net_new.log_prob(sample2)
                        if Tstep==0 and args.IniDistri=='delta': # delta distribution
                            sample2=torch.tensor(np.tile(args.initialD,[args.batch_size,1]),dtype=default_dtype_torch).to(args.device)
                            log_prob0 = torch.zeros(sample2.shape[0]).to(args.device)
                        if Tstep==0 and args.IniDistri=='poisson' and args.Model=='BirthDeath':
                            sampleL=torch.tensor(range(args.M))
                            Sample1D=(sampleL.view(-1, args.size))
                            initialD2=args.initialD.repeat(args.M, axis=0).reshape(-1,1)
                            samplepoisson=torch.as_tensor(poisson.pmf(k=Sample1D.cpu(), mu=initialD2)).repeat(1,args.batch_size).t()
                            values,indices=torch.max(torch.rand(args.batch_size,1).to(args.device)<torch.cumsum(samplepoisson,dim=1),dim=1)
                            sample2=torch.tensor([indices[j] for j in range(args.batch_size)],dtype=default_dtype_torch).view(args.batch_size,args.L).to(args.device).detach()
                            initialD2=args.initialD.repeat(args.batch_size, axis=0).reshape(-1,1)
                            Sample1D=(sample2.view(-1, args.size))
                            log_prob0=torch.log(torch.as_tensor(poisson.pmf(k=Sample1D.cpu(), mu=initialD2),dtype=default_dtype_torch)).to(args.device)
                        
           
                        LogTP_t,delta_T,logPdt=TransitionState4(sample2,args,Tstep,net_new,ReactionMatLeft,V,cc,delta_T,model) #LogTP_t
                        Sample2D=(sample.view(-1, args.size)) #printsample
                        print(torch.mean(Sample2D,0),torch.mean(torch.mean(Sample2D,0)),Tstep)
                

                    log_prob = net.log_prob(sample2) #LogP_t+1
                    if args.lossType=='ForwardKL1' or args.lossType=='ForwardKL2':loss=abs(-torch.mean(log_prob-LogTP_t));#loss= -log_prob+LogTP_t.detach();loss=torch.mean((loss - loss.mean())*log_prob) #
                    if args.lossType=='ForwardKL3':loss=abs(torch.mean(torch.exp(LogTP_t.detach()-log_prob0.detach())*(LogTP_t.detach()-log_prob))) #-torch.sum(log_prob)   
                    if args.lossType=='ReverseKL2':
                        with torch.no_grad():loss0=torch.exp(log_prob.detach()-log_prob0.detach())*(log_prob.detach()-LogTP_t.detach())
                        loss=torch.mean((loss0)*log_prob) 
                    if args.lossType=='ReverseKL3':
                        with torch.no_grad():loss0=torch.exp(log_prob.detach()-LogTP_t.detach())*(log_prob.detach()-LogTP_t.detach())
                        loss=torch.mean((loss0)*log_prob) 
                    loss.backward()           
                        
                    if args.clip_grad:
                        nn.utils.clip_grad_norm_(params, args.clip_grad)           
                    optimizer.step()          
                    if args.lr_schedule:
                        scheduler.step(loss.mean())     
                    train_time += time.time() - train_start_time                    

                    loss_std = loss.std()
                    loss_mean=loss.mean()
                    loss_forwardKL = abs(torch.mean(LogTP_t-log_prob))
                    loss_reverseKL = loss#abs(torch.mean(torch.exp(-log_prob+LogTP_t)*(-log_prob+LogTP_t)))
                    
                    Listloss_mean.append(loss_mean.detach().cpu().numpy())
                    Listloss_std.append(loss_std.detach().cpu().numpy())
                    Listloss_forwardKL.append(loss_forwardKL.detach().cpu().numpy())
                    Listloss_reverseKL.append(loss_reverseKL.detach().cpu().numpy())
                    
                    if step==int(args.max_step):
                        with torch.no_grad():
                            for _  in range(10):
                                sample, x_hat = net.sample(args.batch_size)
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
                        argsSave=[args.Tstep,args.delta_t,args.L,args.M,args.lr,args.initialD,args.print_step]
                        SummaryListDynPartiFuncLog2=np.array(SummaryListDynPartiFuncLog2)
                        SummaryLoss1=[]
                        SummaryLoss2=[]
                        np.savez('{}_img/Data'.format(args.out_filename),SummaryListDynPartiFuncLog2,argsSave,SummaryListDynPartiFuncLog3,SummaryLoss1,SummaryLoss2,np.array(SampleSum),np.array(delta_TSum))     
                    
                    
                        
                # Plot training loss
                if Tstep in args.training_loss_print_step:#[0,1,2,101,1001,2e3,1e4,1e5,2e5,3e5,4e5,5e5]:
                # if Tstep<3 or Tstep==101 or Tstep==1001 or Tstep==2000 or Tstep==10000 or Tstep==100000 or Tstep==200000 or Tstep==300000 or Tstep==400000 or Tstep==500000:
                    plt.figure(num=None,  dpi=300, edgecolor='k')
                    fig, axes = plt.subplots(2,1)
                    fig.tight_layout()
                    ax = plt.subplot(2,1, 1)
                    # plt.plot(range(last_step + 1, args.max_step + 1), Listloss_mean, label='Loss_mean')
                    # plt.plot(range(last_step + 1, args.max_step + 1), Listloss_std, label='Loss_std')
                    plt.plot(range(last_step + 1, args.max_step + 1), Listloss_forwardKL, label='Forward KL')
                    axes = plt.gca()
                    axes.set_yscale('log')
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')
                    plt.title('Time step '+str(Tstep))
                    plt.legend()        
                    ax = plt.subplot(2,1, 2)
                    axes = plt.gca()
                    axes.set_yscale('log')
                    # plt.plot(range(last_step + 1, args.max_step + 1), ListDistanceCheck_Eucli, label='Euclidean distance')
                    plt.plot(range(last_step + 1, args.max_step + 1), Listloss_reverseKL, label='Reverse KL')
                    plt.ylabel('Distance')
                    plt.xlabel('Epoch')
                    fig = plt.gcf()
                    plt.legend()
                    fig.set_size_inches(8, 8)
                    plt.tight_layout()        
                    plt.savefig('{}_img/TimeStep{}.jpg'.format(args.out_filename, Tstep), dpi=300)
                
                Lossmean.append(Listloss_mean)
                Lossstd.append(Listloss_std)

            SummaryLoss1.append(Loss1)      
            SummaryLoss2.append(Loss2) 
            SummaryListDynPartiFuncLog3.append(ListDynPartiFuncLog3)
            np.savez('{}_img/Data'.format(args.out_filename),np.array(SummaryListDynPartiFuncLog2),np.array(SampleSum),np.array(delta_TSum))    
    
    print('SampleSum.shape',np.array(SampleSum).shape)
    end_time2 = time.time()  
    print('Time(min) ',(end_time2-start_time2)/60)
    print('Time(hr) ',(end_time2-start_time2)/3600)
    TimeEnd=np.array((end_time2-start_time2)/3600)
    argsSave=[args.Tstep,args.delta_t,args.L,args.M,args.lr,args.initialD,args.print_step,TimeEnd]
      
    SummaryListDynPartiFuncLog2=np.array(SummaryListDynPartiFuncLog2)
    SummaryLoss1=np.array(SummaryLoss1)
    SummaryLoss2=np.array(SummaryLoss2)
    np.savez('{}_img/Data'.format(args.out_filename),SummaryListDynPartiFuncLog2,argsSave,SummaryListDynPartiFuncLog3,SummaryLoss1,SummaryLoss2,np.array(SampleSum),np.array(delta_TSum),Lossmean,Lossstd)     
    
    return SummaryLoss1, np.array(SampleSum)
























