from args import args
import numpy as np
from main import Test
from main2 import Test2

###Add models----------------------------------
from ToggleSwitch import ToggleSwitch
from EarlyLife import EarlyLife
from Epidemic import Epidemic
from cascade1 import cascade1
from cascade1_inverse import cascade1_inverse
from cascade2 import cascade2
from cascade3 import cascade3
from BirthDeath import BirthDeath
from GeneExp import GeneExp
from AFL import AFL

##Set parameters-------------------------------
###Initialize parameters: otherwise the parameters are specified in init_out_dir-> args.py
args.Model='ToggleSwitch' #Model name
args.L=4#Species number
args.M=int(80) #Upper limit of the molecule number: it is adjustable and can be indicated by doing a few Gillespie simulation. 
args.batch_size=1000 #Number of batch samples
args.Tstep=10# Time step of iterating the chemical master equation
args.delta_t=0.0005 #Time step length of iterating the chemical master equation, depending on the reaction rates
args.Para=1

args.net ='rnn'
args.max_stepAll=5000 # Number of epochs at the first time step (usually larger to ensure the accuracy)
args.max_stepLater=100 # Number of epochs of later time steps
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


print(vars()[args.Model])
model = vars()[args.Model](**vars(args)) 
###Add model command-------------------------------  
# if args.Model=='ToggleSwitch':
#     model = ToggleSwitch(**vars(args))   
# if args.Model=='EarlyLife':
#     model = EarlyLife(**vars(args))      
# if args.Model=='Epidemic':
#     model = Epidemic(**vars(args))   
# if args.Model=='cascade1': 
#     model = cascade1(**vars(args)) 
# if args.Model=='cascade1_inverse':
#     model = cascade1_inverse(**vars(args)) 
# if args.Model=='cascade2':
#     model = cascade2(**vars(args))    
# if args.Model=='cascade3':
#     model = cascade3(**vars(args))    
# if args.Model=='BirthDeath':
#     model = BirthDeath(**vars(args))   
# if args.Model=='GeneExp':
#     model = GeneExp(**vars(args))   
# if args.Model=='AFL':
#     model = AFL(**vars(args)) 
    
#Run model-----------------------------------------        
if __name__ == '__main__':
    
    SummaryLoss={}  
    for args.delta_t in [0.001,0.005,0.01,0.05,0.1]:
        args.net_depth = 1
        args.net_width = 32
        Loss, SampleSum = Test(model)
        SummaryLoss[args.delta_t]=np.mean(Loss)
    print('Best args.delta_t:',min(SummaryLoss.keys(), key=(lambda k: SummaryLoss[k])))
        
     
    SummaryLoss={}    
    for args.net_depth in [1,2]:
        for args.net_width in [8,16,32,64]:
            args.delta_t = 0.005 
            Loss, SampleSum = Test(model) 
            SummaryLoss[(args.net_depth,args.net_width)]=np.mean(Loss)
    print('Best (depth,width):',min(SummaryLoss.keys(), key=(lambda k: SummaryLoss[k])))
    
    
    
    
    
