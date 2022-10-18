from args import args
import numpy as np
from main import Test

###Add models----------------------------------
from ToggleSwitch import ToggleSwitch
from EarlyLife import EarlyLife
from Epidemic import Epidemic
from cascade1 import cascade1

##Set parameters-------------------------------
###Initialize parameters: otherwise the parameters are specified in init_out_dir-> args.py
args.Model='cascade1' #Model name
args.L=15#Species number
args.M=int(80) #Upper limit of the molecule number
args.batch_size=100 #Number of samples
args.Tstep=100# Time step of iterating the dynamical equation P_tnew=T*P_t, where T=(I+W*delta t)
args.delta_t=0.0005 #Time step length of iterating the dynamical equation

args.net ='rnn'
args.max_stepAll=3000 #Maximum number of steps first time step (usually larger to ensure the accuracy)
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
args.print_step=20
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

###Add model command-------------------------------
if args.Model=='ToggleSwitch':
    model = ToggleSwitch(**vars(args))   
if args.Model=='EarlyLife':
    model = EarlyLife(**vars(args))   
if args.Model=='Epidemic':
    model = Epidemic(**vars(args))   
if args.Model=='cascade1': 
    model = cascade1(**vars(args)) 

#Run model-----------------------------------------        
if __name__ == '__main__':
    Test(model)    
    
    
    
    
    
