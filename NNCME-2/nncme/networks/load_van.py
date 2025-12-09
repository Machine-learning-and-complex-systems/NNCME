import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"
import numpy as np
import torch
import copy
from nncme.args import args
from nncme.networks.gru import GRU
from nncme.utils import ensure_dir
import time
from nncme.training.transition_tdvp import TransitionState
from nncme.utils import default_dtype_torch
from nncme.systems.toggle_switch import ToggleSwitch
from nncme.systems.early_life import EarlyLife
from nncme.systems.epidemic import Epidemic
from nncme.systems.cascade1 import cascade1
from nncme.systems.cascade1_inverse import cascade1_inverse
from nncme.systems.cascade2 import cascade2
from nncme.systems.cascade3 import cascade3
from nncme.systems.birth_death import BirthDeath
from nncme.systems.gene_expression import GeneExp
from nncme.systems.afl import AFL

def Optimizer(net,args):

    """Optimizer operation.

    """


    params = list(net.parameters()) ############################
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = int(sum([np.prod(p.shape) for p in params]))
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))

    return optimizer,params,nparams

##Set parameters-------------------------------
###Initialize parameters: otherwise the parameters are specified in init_out_dir-> args.py
args.Model='cascade1' #Model name
args.L=15#Species number
args.Sites=1 # for spatial-extended systems
if args.Sites>1: # for spatial-extended systems
    args.L=args.L*args.Sites
args.M=int(10) #Upper limit of the molecule number: it is adjustable and can be indicated by doing a few Gillespie simulation. 
args.batch_size=10#1000 #Number of batch samples
args.Tstep=1000# Time step of iterating the chemical master equation
args.delta_t=0.005 #Time step length of iterating the chemical master equation, depending on the reaction rates
args.size=args.L

args.net ='rnn'
args.max_stepAll=5000 #Maximum number of steps first time step (usually larger to ensure the accuracy)
args.max_stepLater=100 #Maximum number of steps of later time steps
args.net_depth=1 # including output layer and not input data layer
args.net_width=32
args.d_model=16# transformer
args.d_ff=32# transformer
args.n_layers=2# transformer
args.n_heads=2# transformer
args.lr=0.001
args.binary=False
args.AdaptiveT=False
args.AdaptiveTFold=5
args.print_step=1
args.saving_data_time_step=[0,1e2,5e2,2e3,1e4,2e4,5e4,1e5,1.5e5,2e5,2.5e5,3e5,3.5e5,4e5,5e5,6e5,7e5,8e5,9e5,1e6] #To save data at which time steps (give in a list)
args.training_loss_print_step=[0,1,2,101,1001,2e3,1e4,1e5,2e5,3e5,4e5,5e5] #To print training loss at which time steps (give in a list)

###Default parameters:
args.bias=True
args.bits=1
if args.binary:
    args.bits=int(np.ceil(np.log2(args.M)))
args.Percent=0.2         
args.clip_grad=1
args.dtype='float64'
args.epsilon=1e-30#initial probability of zero species number
args.lr_schedule=False#True


args.loadVAN=True
args.loadTime=500
args.out_filename='cascade115_out'

model = vars()[args.Model](**vars(args))  
args.IniDistri,args.initialD,r,ReactionMatLeft,ReactionMatRight,args.MConstrain,args.conservation=model.rates()
cc=torch.as_tensor(r, dtype=default_dtype_torch).to(args.device)
V=ReactionMatRight-ReactionMatLeft #SpeciesXReactions 
        
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

# SampleT=[]
# start=time.time()

# sample, x_hat = net.sample(args.batch_size)
# SampleT.append(np.array(sample.detach().cpu()))         
# end=time.time()
# print('VAN time(min):',(end-start)/60) 
    
# SampleT=np.array(SampleT).reshape(-1,args.L)

# log_prob = net.log_prob(sample)
# prob = torch.exp(log_prob).detach().numpy()

# out_filename='cascade115_outTstep500_sample100000'
# # np.savez('{}'.format(out_filename),SampleT,prob)    


#Train VAN
sample, x_hat = net.sample(args.batch_size)
optimizer,params,nparams = Optimizer(net,args)

Fk=torch.empty(nparams,1)
Skk=torch.empty(nparams,nparams)
Tstep=0
step=1
delta_T=1
LogTP_t,delta_T,logPdt=TransitionState(sample,args,Tstep,step,net_new,ReactionMatLeft,V,cc,delta_T,model)#.detach()

for i in range(args.batch_size):######################
      
    log_prob = net.log_prob(sample[i].reshape(1,args.L)) #sample has size batchsize X 1 X systemSize
    prob = torch.exp(log_prob)                


    log_prob.backward()
    params_grad=torch.tensor([])
    params2=torch.tensor([])
    for ii in params:##########################
        params2=torch.cat((params2,ii.flatten()))
        params_grad=torch.cat((params_grad,ii.grad.flatten()))
    
    logP_theta=params_grad 

    
    # logPdt=1+(torch.sum(torch.exp(LogP_t_other-LogP_t.repeat(cc.shape[0],1).t())*Propensity_in,1)-R)*args.delta_t
    logP_dt=logPdt[i]/args.delta_t #logP/dt
    
    Fa=(prob*logP_dt*logP_theta).reshape(nparams,1) #params x 1
    Sa=prob*torch.mm(logP_theta.reshape(nparams,1),logP_theta.reshape(1,nparams))

    Fk=Fk+Fa
    Skk=Sa+Sa
    
    optimizer.zero_grad() 
    
dtheta_dt=torch.mm(Skk.t(),Fk)
print(params2[-5:-1])
# params2=params2.reshape(nparams,1)
# print(params2.shape,(dtheta_dt*args.delta_t).shape)
params2=torch.add(params2.reshape(nparams,1),dtheta_dt*args.delta_t)
print(params2[-5:-1])
dtheta=dtheta_dt*args.delta_t
params2=params2.reshape(nparams)

print(params[5])
# print(list(net.parameters())[5])
i=0
for ii in range(len(params)):
    j=i+int(np.prod(params[ii].shape))
    params[ii].data=params[ii]+dtheta[i:j].reshape(params[ii].shape)
    i=j
print(params[5])
# print(list(net.parameters())[5])
        
        

