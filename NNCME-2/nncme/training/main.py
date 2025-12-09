"""SGD, Natural gradient and TDVP training loop for CME models."""

import copy
import math
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import poisson, zipf
from torch import nn
from torch.func import functional_call, grad, vmap
from tqdm import tqdm

from nncme.args import args
from nncme.networks.gru import GRU
from nncme.networks.nade import NADE
from nncme.networks.transformer import TraDE
from nncme.training.transition_tdvp import DeltaFunction, TransitionState
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
from nncme.utils import cholesky_solve, cholesky_solve_fast, default_dtype_torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
plt.rc("font", size=16)

def adaptive_ylim(y_max):
    """
    Set y-axis limit for plots based on the maximum value.
    Args:
        y_max (float): Maximum value in the data.
    Returns:
        float: Suggested upper limit for y-axis.
    """
    if y_max >= 0.8:  
        return 1.0
    elif y_max >= 0.4:
        return 0.8
    elif y_max >= 0.05:
        return 0.4
    else:
        return 0.05

def adaptive_alpha(t):
    """
    Adaptively set the alpha parameter for sampling based on t.
    Args:
        t (float): Current time or normalized time.
    Returns:
        float: Alpha value.
    """
    if t <= 0.2: return 0.2
    if t <= 0.6:  return 0.3
    return 0.3


def Optimizer(net,args):

    """
    Set up the optimizer for the neural network.
    Args:
        net (nn.Module): Neural network model.
        args: Global arguments, must have lr.
    Returns:
        optimizer: Adam optimizer instance.
        params: List of trainable parameters.
        nparams: Total number of trainable parameters.
    """
    params = list(net.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = int(sum([np.prod(p.shape) for p in params]))
    my_log('Total number of trainable parameters: {}'.format(nparams))
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    return optimizer,params,nparams

                   
def TestNatGrad(model):    
    """
    Main training and inference loop using Natural Gradient or TDVP for CME solution.
    Args:
        model: Model object, must have rates() and Propensity().
    Returns:
        Loss1 (list): Loss mean at each print step.
        SampleSum (ndarray): All samples at each print step, shape (num_prints, batch, L).
        delta_TSum (ndarray): Time step size at each print step.
        Lossmean (list): All loss means during training.
        Lossstd (list): All loss stds during training.
        argsSave (list): Saved arguments and timing info.
    """
    
    start_time2 = time.time()    
        
    if args.Model!='Epidemic':
        args.IniDistri,args.initialD,r,ReactionMatLeft,ReactionMatRight,args.MConstrain,args.conservation=model.rates()
        cc=torch.as_tensor(r, dtype=default_dtype_torch).to(args.device)
        V=ReactionMatRight-ReactionMatLeft #SpeciesXReactions 
        # my_log('r {}'.format(r))
        # my_log('ReactionMatLeft {}'.format(ReactionMatLeft))
        # my_log('ReactionMatRight {}'.format(ReactionMatRight))
    args.size=args.L # the number of spin: 1D, doesnt' count the boundary spins
    
    start_time = time.time()
    init_out_dir()
    if args.clear_checkpoint:
        clear_checkpoint()
    last_step = get_last_checkpoint_step()
    my_log(str(last_step))
    if last_step >= 0:
        my_log('\nCheckpoint found: {}\n'.format(last_step))
    else:
        clear_log()    
    my_log(args.out_filename)    

    my_log('args.initialD {}'.format(args.initialD))
    my_log('args.conservation {}'.format(args.conservation))
    my_log('args.MConstrain {}'.format(args.MConstrain))
    my_log('args.cuda {}'.format(args.cuda))
    my_log('args.device {}'.format(args.device))
    my_log('args.L_label {}'.format(args.L_label))
    my_log('args.L_plot {}'.format(args.L_plot))
    my_log('args.sampling {}'.format(args.sampling))
    my_log('args.absorbed {}'.format(args.absorbed))
    my_log('args.absorb_state {}'.format(args.absorb_state))
    

    args.print_step = max(10, int(args.Tstep / args.num_prints)) 
    args.plotstep = max(10, int(args.Tstep / args.num_plots)) 
    args.absorb_step = max(10, int(args.Tstep / args.num_absorbs))
    my_log('args.print_step {}'.format(args.print_step))
    my_log('args.plotstep {}'.format(args.plotstep))
    my_log('args.absorb_step {}'.format(args.absorb_step))
    
    net_new = []
    loss=[1]
    for lambda_tilt in np.arange(1):
        args.lambda_tilt=lambda_tilt
        args.max_step=args.epoch0
        #Initialize net and optimizer  
        if args.net == 'rnn':
            net = GRU(**vars(args))
        elif args.net == 'transformer':
            net = TraDE(**vars(args)).to(args.device)    
        elif args.net == 'NADE':
            net = NADE(**vars(args)).to(args.device) 
        else:
            raise ValueError('Unknown net: {}'.format(args.net))
        net.to(args.device)
        

        Loss1=[] # loss_mean at the end of every Tstep
        Loss2=[] # loss_std at the end of every Tstep
        SampleSum=[]
        delta_TSum=[]
        Lossmean=[] # all loss_mean in epochs and Tsteps
        Lossstd=[] # all loss_std in epochs and Tsteps
        nets_dict = {}
        
        #Load NN starting from certain time point:
        startT=0
        if args.loadVAN:
            PATH='\\nd1_nw8_NADE_NatGrad_lr0.5_epoch5_Losskl_Samplingrandom200_IniDistdelta_Para1.0_bias_cg1\\outTstep'+str(args.loadTime) # Change the load VAN path
            startT=args.loadTime
            print(PATH)
            if args.cuda==-1:
                state = torch.load(PATH,map_location=torch.device('cpu') ) #CPU  
            else:
                state = torch.load(PATH) #GPU
            # net1=NADE(**vars(args))#GRU(**vars(args))
            if args.net == 'rnn':
                net1 = GRU(**vars(args))
            elif args.net == 'transformer':
                net1 = TraDE(**vars(args)).to(args.device)    
            elif args.net == 'NADE':
                net1 = NADE(**vars(args)).to(args.device) 
            else:
                raise ValueError('Unknown net: {}'.format(args.net))
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
        if args.absorbed:
            absorbs_time=[]
            absorbs_prob=[]
        
        tolerance = 1e-3
        stop_condition_met = False
        
        for Tstep in tqdm(range(startT, args.Tstep), desc="Training Tstep", ncols=100, mininterval=600):
            if args.alpha<=0:
                args.alpha=adaptive_alpha(Tstep*args.delta_t)
                
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
            if stop_condition_met:
                break
              
            
            ListDistanceCheck_Eucli=[]
            ListDistanceCheck=[]
            Listloss_mean=[]
            Listloss_std=[]
            if Tstep>=1:
                args.max_step=args.epoch
                if args.loadVAN and Tstep==startT:
                    args.max_step=args.epoch0

            # %% SGD
            if args.method=='SGD':#if Tstep<10000:#==0:#<20:
                SampleT=[]
                for step in range(last_step + 1, args.max_step + 1):    
                    optimizer.zero_grad()   
                    with torch.no_grad():
                        if args.sampling=='diffusive' and Tstep != 0:
                            sample2, x_hat = net.sampleDiffusive(args.batch_size)
                            sample = sample2.detach()
                        elif args.sampling=='alpha' and Tstep != 0:
                            sample2, x_hat = net.sampleAlpha(args.batch_size)
                            sample = sample2.detach()
                        else:
                            sample, x_hat = net.sample(args.batch_size)
                        if args.sampling=='diffusive':
                            log_prob_conv = net.log_prob_diff(sample) 
                        elif args.sampling=='alpha':
                            log_prob_conv = net.log_prob_alpha(sample) 
                    log_prob = net.log_prob(sample) #sample has size batchsize X 1 X systemSize
                    if Tstep % int(args.print_step)==0:
                        if step%50==0:# For my computer
                            my_log('{} {} {}'.format(torch.mean(sample,0),Tstep,step))
                    if step==0: delta_T=1
                    with torch.no_grad():
                        if args.sampling=='diffusive' or args.sampling=='alpha':
                            if Tstep == 0:
                                LogTP_t,delta_T,logPdt=TransitionState(sample,args,Tstep,step,net_new,ReactionMatLeft,V,cc,delta_T,model)#.detach()
                                loss = log_prob-LogTP_t.detach()
                            else:
                                LogTP_t,delta_T,logPdt=TransitionState(sample,args,Tstep,step,net_new,ReactionMatLeft,V,cc,delta_T,model)#.detach()
                                loss = torch.exp(log_prob-log_prob_conv.detach())*(log_prob-LogTP_t.detach())
                        else:
                            LogTP_t,delta_T,logPdt=TransitionState(sample,args,Tstep,step,net_new,ReactionMatLeft,V,cc,delta_T,model)#.detach()
                            TP_t_normalize=(torch.exp(LogTP_t)/(torch.exp(LogTP_t)).sum()*(torch.exp(log_prob)).sum()).detach()
                            loss= log_prob-LogTP_t.detach()             
                            lossL2 = (torch.exp(log_prob)-TP_t_normalize)
                            losshe = -torch.sqrt(torch.exp(log_prob)*TP_t_normalize)
                    
                    assert not LogTP_t.requires_grad
                    if args.lossType=='kl':
                        loss_reinforce = torch.mean((loss - loss.mean())*log_prob)       
                        if args.sampling=='diffusive' or args.sampling=='alpha':
                            loss_reinforce = torch.mean((loss)*log_prob)
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
                    
                    loss_std = loss.std()
                    loss_mean=abs(loss.mean())
                    Listloss_mean.append(loss_mean.detach().cpu().numpy())
                    Listloss_std.append(loss_std.detach().cpu().numpy())
            elif args.method=='NatGrad' or args.method=='TDVP':
                #Train VAN
                SampleT=[]
                epoch=args.epoch#5
                if Tstep<=0:epoch=args.epoch0#50
                for step in range(epoch):
                    if step==0:
                        delta_T=1
                    with torch.no_grad():
                        weights = torch.ones(args.batch_size, device=args.device) 

                        if args.sampling=='default':
                            sample, x_hat = net.sample(args.batch_size)
                            log_prob = net.log_prob(sample) #sample has size batchsize X 1 X systemSize         
                            LogTP_t,delta_T,logPdt=TransitionState(sample,args,Tstep,step,net_new,ReactionMatLeft,V,cc,delta_T,model)#.detach()
                            loss = log_prob-LogTP_t.detach()
                            
                        elif args.sampling == 'random':
                            N1, N2 = args.batch_size - args.ESNumber, args.ESNumber
                            sample1, _ = net.sample(N1)
                            if args.MConstrain[0] == 0:
                                sample2 = torch.randint(0, args.M, (N2, sample1.shape[1]), device=args.device)
                                Mi = torch.full((sample1.shape[1],), args.M, device=args.device, dtype=torch.float32)
                            else:
                                sample2 = torch.zeros((N2, sample1.shape[1]), dtype=torch.long, device=args.device)
                                Mi = torch.tensor([args.MConstrain[i] for i in range(sample1.shape[1])],
                                                  device=args.device, dtype=torch.float32)
                                for i in range(sample1.shape[1]):
                                    sample2[:, i] = torch.randint(0, args.MConstrain[i], (N2,), device=args.device)
                            sample = torch.cat([sample1, sample2], dim=0).detach()
                            log_prob = net.log_prob(sample)
                            LogTP_t, delta_T, logPdt = TransitionState(sample,args,Tstep,step,net_new,ReactionMatLeft,V,cc,delta_T,model)
                            loss = log_prob - LogTP_t.detach()
                            
                            if args.reweighted and N2 > 0:
                                log_u = -torch.log(Mi).sum()
                                weights[N1:] = torch.exp((log_prob[N1:] - log_u).detach())
                                weights = weights / (weights.mean() + 1e-8)
                            
                        elif args.sampling in ['diffusive', 'alpha']:
                            if Tstep == 0:
                                sample, _ = net.sample(args.batch_size);
                                log_prob = net.log_prob(sample)
                                LogTP_t, delta_T, logPdt = TransitionState(sample,args,Tstep,step,net_new,ReactionMatLeft,V,cc,delta_T,model)
                                loss = log_prob - LogTP_t.detach()
                            else:
                                if args.sampling == 'diffusive':
                                    sample, _ = net.sampleDiffusive(args.batch_size) 
                                    log_prob_conv = net.log_prob_diff(sample) 
                                else:
                                    sample, _ = net.sampleAlpha(args.batch_size)
                                    log_prob_conv = net.log_prob_alpha(sample) 
                                sample = sample.detach()
                            
                                log_prob = net.log_prob(sample)
                                LogTP_t, delta_T, logPdt = TransitionState(sample,args,Tstep,step,net_new,ReactionMatLeft,V,cc,delta_T,model)
                                # LogTP_t ： target prob
                                if args.reweighted:
                                    logw = (log_prob - log_prob_conv).detach()
                                    weights, k_hat = psis_weights(logw, tail_frac=0.2) 
                                    gamma = 0.2
                                    N = weights.numel()
                                    uniform = torch.full_like(weights, 1.0 / N)
                                    weights = (1 - gamma) * weights + gamma * uniform
                                    weights = weights / (weights.mean())
                                loss = log_prob - LogTP_t.detach()

                    grads = net.per_sample_grad(sample) 
                    grads_flatten = torch.cat([torch.flatten(v, start_dim=1) for v in grads.values()], dim=1)
                    N = sample.size(0)
                    eps = 1e-12


                    O_mat = (torch.sqrt(weights + eps).unsqueeze(1) * grads_flatten) / math.sqrt(N)
                    
                    if Tstep == 0 or args.method=='NatGrad':
                        baseline = (weights * loss).mean() #loss.mean()
                        F_vec = torch.einsum("nm,n->m", grads_flatten, weights * (loss - baseline)) / N
                        updates_flatten = cholesky_solve_fast(O_mat, F_vec)
                        net.update_params(updates_flatten, args.lr)
                    elif args.method=='TDVP':
                        target = (-logPdt)
                        base_t = (weights * target).mean()
                        F_vec = torch.einsum("nm,n->m", grads_flatten, weights * (target-base_t)) / N
                        updates_flatten = cholesky_solve_fast(O_mat, F_vec)
                        net.update_params(updates_flatten, args.lr)
                    
                    loss_std = loss.std()
                    loss_mean=loss.mean()
                    Listloss_mean.append(abs(loss_mean).detach().cpu().numpy())
                    Listloss_std.append(loss_std.detach().cpu().numpy())
                    

                    if Tstep==args.Tstep-1:#[0,1e2,5e2,2e3,1e4,2e4,5e4,1e5,1.5e5,2e5,2.5e5,3e5,3.5e5,4e5,5e5,6e5,7e5,8e5,9e5,1e6]:
                        PATH=args.out_filename+'Tstep'+str(Tstep)+'epoch'+str(step)
                        #torch.save(net, PATH)
                        state = {'net': net.state_dict(),
                                'optimizer': optimizer.state_dict()}
                        torch.save(state, PATH)

            # %% Save Sample at the end of each Tstep
            SampleT, x_hat = net.sample(args.save_sample_num)
            if Tstep % int(args.print_step)==0:
                my_log('{} {} {}'.format(torch.mean(SampleT,0),Tstep,step))
                if args.absorbed==True:
                    absorb_min, absorb_max = args.absorb_state
                    absorb_min = torch.tensor(absorb_min).repeat(args.L).to(args.device)
                    absorb_max = torch.tensor(absorb_max).repeat(args.L).to(args.device)
                    count = (SampleT >= absorb_min.unsqueeze(0)).all(dim=1) & (SampleT <= absorb_max.unsqueeze(0)).all(dim=1)           # (batch,)                
                    absorb_p=count.sum()/SampleT.shape[0]
                    my_log('******** P(absorb={}):[{:.3e}]'.format(args.absorb_state,float(absorb_p)) )
               
            if args.absorbed==True:    
               if Tstep % int(args.absorb_step)==0:
                    absorb_min, absorb_max = args.absorb_state
                    absorb_min = torch.tensor(absorb_min).repeat(args.L).to(args.device)
                    absorb_max = torch.tensor(absorb_max).repeat(args.L).to(args.device)
                    count = (SampleT >= absorb_min.unsqueeze(0)).all(dim=1) & (SampleT <= absorb_max.unsqueeze(0)).all(dim=1)           # (batch,)                
                    absorb_p=count.sum()/SampleT.shape[0]
                    absorbs_time.append(Tstep*args.delta_t)
                    absorbs_prob.append(float(absorb_p))

            
            with torch.no_grad():
                if Tstep % int(args.print_step)==0: #if Tstep % (1/args.delta_t)==0: 
                    Loss1.append(np.array(abs(loss_mean).detach().cpu()))      
                    Loss2.append(np.array(loss_std.detach().cpu()))
                    SampleSum.append(np.array(SampleT.detach().cpu()).reshape(-1,args.L))
                    Lossmean.append(Listloss_mean)
                    Lossstd.append(Listloss_std)  
                    try:
                        # init metadata once for reconstruction
                        if '_meta' not in nets_dict:
                            nets_dict['_meta'] = {
                                'net_type': args.net,
                                'args': vars(args),
                                'created_at': time.time(),
                            }
                        tkey = str(int(Tstep))
                        nets_dict[tkey] = {
                            'net': copy.deepcopy(net).cpu().state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }
                    except Exception as e:
                        my_log(f"WARN: failed to checkpoint at Tstep {Tstep}: {e}")
                
                delta_TSum.append(delta_T)
                net_new=copy.deepcopy(net)
                net_new.requires_grad=False

            L_label=args.L_label#=['$m_1$','$p_1$','$m_2$','$p_2$','$m_3$','$p_3$','$n_1$','$n_2$','$n_3$']
            colors=['C0','C1','C2','C3','C4','C5','C6','C7','C8']
            L_plot=args.L_plot#=[0,1,6]
            numplot=len(L_plot)+2
            if Tstep in range(0,args.Tstep,args.plotstep):
                # Tstep=Tstep-args.loadTime
                if args.loadVAN==True:
                    index=int((Tstep-args.loadTime)/args.print_step)
                else:
                    index=int((Tstep)/args.print_step)

                t=Tstep*args.delta_t
                # plt.figure(num=None,  dpi=300, edgecolor='k');
                fig, axes = plt.subplots(1,numplot)
                fig.tight_layout()   
                for j, ii in enumerate(L_plot):
                    ax = plt.subplot(1,numplot, j+1)
                    if args.Model=='BirthDeath':
                        mu0=1
                        mu = mu0*np.exp(-0.01*t)+10*(1-np.exp(-0.01*t))
                        prob_dist = poisson.pmf(range(0,30),mu)
                        plt.plot(prob_dist,'r',label='Analytical')
                    arr=np.array(SampleSum)[index,:,ii]#np.array(SampleT).reshape(-1,args.L)[:,ii]
                    if args.MConstrain[0]!=0: 
                        lim=int(args.MConstrain[ii]/1.2)+10
                    else:lim=int(args.M)#*4/5)
                    hist_values, bin_edges, _ = plt.hist(arr, density=True, color=colors[j], range=(0, lim), bins=lim, label=L_label[j])
                    plt.ylabel('Prob')
                    plt.xlabel('Count')
                    plt.title('t='+str(t))
                    y_max = max(hist_values) if len(hist_values) > 0 else 0
                    y_upper = adaptive_ylim(y_max)
                    plt.ylim(0, y_upper)
                    plt.yticks(np.linspace(0,y_upper,5))
                    fig = plt.gcf()
                    plt.legend(title=f'{args.sampling}')
                
                if args.L>1:
                    ax = plt.subplot(1,numplot, numplot-1)
                    arr1=np.array(SampleSum)[index,:,0]
                    arr2=np.array(SampleSum)[index,:,1]
                    plt.hist2d(arr1,arr2,bins=[int(args.M/2),int(args.M/2)],density=True,norm=mpl.colors.LogNorm(vmin=1e-4,vmax=1e-1))
                    plt.ylabel('X1')
                    plt.xlabel('X2')
                    plt.xlim([0,args.M])
                    plt.ylim([0,args.M])
                    ax.set_facecolor([68/255,1/255,80/255])
                    plt.title('t='+str(t))
                    
                ax = plt.subplot(1,numplot, numplot)
                plt.plot(range(len(Lossmean[index])), Lossmean[index], label='Loss_mean',color='dimgrey')
                plt.plot(range(len(Lossstd[index])), Lossstd[index], label='Loss_std',color='darkgrey')
                axes = plt.gca()
                axes.set_yscale('log')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.title('Tstep='+str(Tstep))
                # plt.ylim(1e-5, 1e1)
                leg = plt.legend(title=f'{args.method}\nepoch={args.epoch} lr={args.lr}')
                leg.get_title().set_ha("center")
                fig.set_size_inches(numplot*5, 5)
                plt.tight_layout()        
                plt.savefig('{}_img/TimeStep{}.jpg'.format(args.out_filename, Tstep), dpi=300)
                plt.show()
                plt.close()
        
        if args.absorbed==True:  
            plt.figure(figsize=(6,4))
            plt.plot(absorbs_time, absorbs_prob, marker='o')
            plt.xlabel("T")
            plt.ylabel("Probability")
            plt.title("X={} to X={}".format(args.initialD[0],args.absorb_state))
            plt.grid(True)
            plt.savefig('{}_img/absorb.jpg'.format(args.out_filename), dpi=300)
            plt.show()
        
        
        data_to_save = {
            "argsSave": [args.Tstep,args.delta_t,args.L,args.M,args.lr,args.initialD,args.print_step,(time.time()-start_time2)/3600],
            "Lossmean": Lossmean,
            "Lossstd": Lossstd,
            "SampleSum": np.array(SampleSum),
            "delta_TSum":np.array(delta_TSum),
        }

        np.save('{}_img/Data.npy'.format(args.out_filename), data_to_save)

        try:
            torch.save(nets_dict, f"{args.out_filename}_img/nets_dict.pt")
            my_log('Saved nets_dict to {}_img/nets_dict.pt'.format(args.out_filename))
        except Exception as e:
            my_log(f"WARN: failed to save nets_dict: {e}")

        if args.absorbed:
            absorb = {
                "absorbs_time": absorbs_time,
                "absorbs_prob": absorbs_prob,
            }
            np.save('{}_img/absorb.npy'.format(args.out_filename), absorb)
        
    my_log('SampleSum.shape {}'.format(np.array(SampleSum).shape))
    end_time2 = time.time()  
    my_log('Time(min) {:.3f}'.format((end_time2-start_time2)/60))
    my_log('Time(hr) {:.3f}'.format((end_time2-start_time2)/3600))
    TimeEnd=np.array((end_time2-start_time2)/3600)
    argsSave=[args.Tstep,args.delta_t,args.L,args.M,args.lr,args.initialD,args.print_step,TimeEnd]
     
    # np.savez('{}_img/Data'.format(args.out_filename),*argsSave,np.array(SampleSum),np.array(delta_TSum),*Lossmean,*Lossstd)     
    with open('data_path.txt', 'w') as file:
        file.write('{}_img'.format(args.out_filename))
    my_log(args.out_filename)
    


    if Tstep==args.Tstep-1 and args.Model in ['Schlogl','Schlogl_2d']:
        with torch.no_grad():
            N = int(1e7)
            B = 100000
            samples = []

            for _ in range(N // B):
                sample_batch, _ = net.sample(B)
                samples.append(sample_batch.cpu().numpy())

            samples_all = np.concatenate(samples, axis=0)  # shape: (1e6, L)

        np.save("sample.npy", samples_all)
        # print("Done. Shape:", samples_all.shape)


        sample = samples_all  # shape (N, L)

        A = [25, 85]
        B = [0, 15]
        in_A = ((sample >= A[0]) & (sample <= A[1])).all(axis=1)
        in_B = ((sample >= B[0]) & (sample <= B[1])).all(axis=1)

        PA = in_A.sum() / sample.shape[0]
        PB = in_B.sum() / sample.shape[0]
        P = PB / PA if PA > 0 else np.inf
        my_log('\n')
        my_log(f"PA = {PA:.3e}, PB = {PB:.3e}, PB/PA = {P:.3e}")



    return Loss1, np.array(SampleSum),np.array(delta_TSum),Lossmean,Lossstd, argsSave



def psis_weights(logw, tail_frac=0.2, eps=1e-12):
    """
    Pareto Smoothed Importance Sampling (PSIS) for reweighting samples.
    Args:
        logw (Tensor): Log importance weights, shape (N,).
        tail_frac (float): Fraction of largest weights to smooth (default=0.2).
        eps (float): Small value for numerical stability.
    Returns:
        w_final (Tensor): Smoothed and normalized weights, shape (N,).
        k_hat (float): Estimated shape parameter for the tail.
    """
    N = logw.shape[0]
    w = torch.exp(logw - torch.max(logw))  # Prevent overflow
    w = w / (torch.sum(w) + eps)           # normalize to sum=1
    # 1. sort weights
    w_sorted, idx = torch.sort(w, descending=True)
    M = max(1, int(tail_frac * N))  # number of tail samples
    tail_w = w_sorted[:M]
    body_w = w_sorted[M:]
    # 2. fit generalized Pareto distribution (GPD) to tail
    # use simple Hill estimator for shape parameter k
    xm = tail_w.min()
    log_ratios = torch.log(tail_w / xm + eps)
    k_hat = (log_ratios.mean() + eps) ** -1   # rough estimator
    k_hat = torch.clamp(k_hat, min=-0.5, max=1.0)  # clip for stability
    # 3. smooth tail weights: replace with expected values under fitted GPD
    j = torch.arange(1, M+1, device=w.device).float()
    smoothed_tail = xm * ( (M / j) ** k_hat )
    smoothed_tail = smoothed_tail / smoothed_tail.sum() * tail_w.sum()
    # 4. reconstruct smoothed weights
    w_smoothed = torch.cat([smoothed_tail, body_w], dim=0)
    # restore original ordering
    w_final = torch.zeros_like(w)
    w_final[idx] = w_smoothed
    # normalize
    w_final = w_final / (w_final.sum() + eps)
    return w_final, k_hat.item()


