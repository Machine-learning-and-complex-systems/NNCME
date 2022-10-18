import argparse
import numpy as np
parser = argparse.ArgumentParser()

group = parser.add_argument_group('physics parameters')
# group.add_argument(
#     '--ham',
#     type=str,
#     default='fm',
#     choices=['afm', 'fm'],
#     help='Hamiltonian model')

group.add_argument(
    '--Model',
    type=str,
    default='No',
    #choices=['No','cascade1','ToggleSwitch','cascade2','cascade3','repressilator','homo1','MM','AFL','GeneExp1', 'GeneExp2', 'BirthDeath', 'Moran','Epidemic'],
    help='Models for master equation')

group.add_argument(
    '--IniDistri',
    type=str,
    default='delta',
    #choices=['delta','poisson'],
    help='Initial Distribution for the species')


group.add_argument('--Tstep', type=int, default=1, help='Time step of iterating the dynamical equation P_tnew=T*P_t')
group.add_argument('--Para', type=float, default=1, help=' parameter for the model')
group.add_argument(
    '--loadTime', type=int, default=1000, help='loadTime')
group.add_argument('--delta_t', type=float, default=0.05, help='Time step length of iterating the dynamical equation')
group.add_argument('--AdaptiveTFold', type=float, default=100, help='Explore the Adaptive T increase Fold')
#group.add_argument('--c', type=float, default=0.5, help='Flip-up probability')
# group.add_argument('--dlambda', type=float, default=0.2, help='steplength for the tilt parameter')
# group.add_argument('--dlambdaL', type=float, default=-2, help='10^L: Left bounary of scan for the tilt parameter')
# group.add_argument('--dlambdaR', type=float, default=0, help='10^R: Right bounary of scan for the tilt parameter')
group.add_argument(
    '--lr_schedule_type',
    type=int,
    default=1,
    help='lr rate schedulers')

group.add_argument(
    '--lossType',
    type=str,
    default='kl',
    choices=['l2','kl', 'he','ss'],
    help='Loss functions: l2, KL-divergence, and Hellinger')

# group.add_argument(
#     '--lattice',
#     type=str,
#     default='sqr',
#     choices=['sqr', 'tri'],
#     help='lattice shape')
group.add_argument(
    '--boundary',
    type=str,
    default='periodic',
    choices=['open', 'periodic'],
    help='boundary condition')
group.add_argument(
    '--L',
    type=int,
    default=3,
    help='number of sites on each edge of the lattice')
group.add_argument(
    '--order',
    type=int,
    default=1,
    help='forward or reverse order for species')

group.add_argument(
    '--M',
    type=int,
    default=1e2,
    help='Upper limit of the Molecule number')



group.add_argument('--beta', type=float, default=1, help='beta = 1 / k_B T')
group.add_argument('--Percent', type=float, default=0.1, help='Default 0.1; Percent: 1. Percent of laster epochs used to calculate free energy; 2. Percent of epochs to reduce lr')

group = parser.add_argument_group('network parameters')
group.add_argument(
    '--net',
    type=str,
    default='rnn',
    choices=['made','rnn', 'transformer'],
    help='network type')
group.add_argument('--net_depth', type=int, default=3, help='network depth')
group.add_argument('--net_width', type=int, default=64, help='network width')
group.add_argument('--d_model', type=int, default=64, help='transformer')
group.add_argument('--d_ff', type=int, default=128, help='transformer')
group.add_argument('--n_layers', type=int, default=2, help='transformer')
group.add_argument('--n_heads', type=int, default=2, help='transformer')
group.add_argument(
    '--half_kernel_size', type=int, default=1, help='(kernel_size - 1) // 2')
group.add_argument(
    '--dtype',
    type=str,
    default='float32',
    choices=['float32', 'float64'],
    help='dtype')
group.add_argument('--bias', action='store_true', help='use bias')
group.add_argument('--AdaptiveT', action='store_true', help='use AdaptiveT')
group.add_argument('--binary', action='store_true', help='use binary conversion')
#group.add_argument('--conservation', action='store_true', help='imposing conservation of some quantities')
group.add_argument('--conservation', type=int, default=1, help='imposing conservation of some quantities')
group.add_argument('--reverse', action='store_true', help='with reverse conditional probability')
group.add_argument(
    '--loadVAN', action='store_true', help='load VAN at later time points')
# group.add_argument(
#     '--z2', action='store_true', help='use Z2 symmetry in sample and loss')
group.add_argument('--res_block', action='store_true', help='use res block')
# group.add_argument(
#     '--x_hat_clip',
#     type=float,
#     default=0,
#     help='value to clip x_hat around 0 and 1, 0 for disabled')
# group.add_argument(
#     '--final_conv',
#     action='store_true',
#     help='add an additional conv layer before sigmoid')
group.add_argument(
    '--epsilon',
    type=float,
    default=0,#default=1e-39,  
    help='small number to avoid 0 in division and log')

group.add_argument(
    '--initialD',
    type=float,
    default=1,#default=1e-39,  
    help='the parameter for the initial Poisson distribution')
group.add_argument(
    '--MConstrain',
    type=int,
    default=np.zeros(1,dtype=int),#default=1e-39,  
    help='MConstrain')


group = parser.add_argument_group('optimizer parameters')
group.add_argument(
    '--seed', type=int, default=0, help='random seed, 0 for randomized')
group.add_argument(
    '--optimizer',
    type=str,
    default='adam',
    choices=['sgd', 'sgdm', 'rmsprop', 'adam', 'adam0.5'],
    help='optimizer')
group.add_argument(
    '--batch_size', type=int, default=10**3, help='number of samples')
group.add_argument('--lr', type=float, default=1e-3, help='learning rate')
group.add_argument(
    '--max_step', type=int, default=10**3, help='maximum number of steps')
group.add_argument(
    '--max_stepAll', type=int, default=10**4, help='maximum number of steps')
group.add_argument(
    '--max_stepLater', type=int, default=50, help='maximum number of steps of later time step')
group.add_argument(
    '--lr_schedule', action='store_true', help='use learning rate scheduling')
group.add_argument(
    '--beta_anneal',
    type=float,
    default=0,
    help='speed to change beta from 0 to final value, 0 for disabled')
group.add_argument(
    '--clip_grad',
    type=float,
    default=0,
    help='global norm to clip gradients, 0 for disabled')

group = parser.add_argument_group('system parameters')
group.add_argument(
    '--no_stdout',
    action='store_true',
    help='do not print log to stdout, for better performance')
group.add_argument(
    '--clear_checkpoint', action='store_true', help='clear checkpoint')
group.add_argument(
    '--print_step',
    type=int,
    default=100,
    help='number of steps to print log, 0 for disabled')
group.add_argument(
    '--save_step',
    type=int,
    default=100,
    help='number of steps to save network weights, 0 for disabled')
group.add_argument(
    '--visual_step',
    type=int,
    default=100,
    help='number of steps to visualize samples, 0 for disabled')
group.add_argument(
    '--save_sample', action='store_true', help='save samples on print_step')
group.add_argument(
    '--print_sample',
    type=int,
    default=1,
    help='number of samples to print to log on visual_step, 0 for disabled')
group.add_argument(
    '--print_grad',
    action='store_true',
    help='print summary of gradients for each parameter on visual_step')
group.add_argument(
    '--cuda', type=int, default=-1, help='ID of GPU to use, -1 for disabled')
group.add_argument(
    '--out_infix',
    type=str,
    default='',
    help='infix in output filename to distinguish repeated runs')
group.add_argument(
    '-o',
    '--out_dir',
    type=str,
    default='out',
    help='directory prefix for output, empty for disabled')


group.add_argument(
    '--species_num',
    type=int,
    default=3,
    help='Number of species in the system')
group.add_argument(
    '--upper_limit',
    type=int,
    default=1e2,
    help='Upper limit of the Molecule number')
group.add_argument(
    '--reaction_num',
    type=int,
    default=3,
    help='Number of reactions in the system')
group.add_argument(
    '--reaction_rates',
    type=list,
    default=[0],
    help='Listed rates for each reaction')
group.add_argument(
    '--initial_distribution',
    type=str,
    default='delta',
    choices=['delta','poisson'],
    help='Initial Distribution for the species')
group.add_argument(
    '--initial_num',
    type=list,
    default=[0],
    help='Initial number for each species in a list')
group.add_argument(
    '--reaction_matrix_left',
    type=list,
    default=[0],
    help='The reaction matrix on the left (dimension: species_num*reactions_num) (stoichiometric matrix = reaction_matrix_left-reaction_matrix_right)')
group.add_argument(
    '--reaction_matrix_right',
    type=list,
    default=[0],
    help='The reaction matrix on the right (dimension: species_num*reactions_num) (stoichiometric matrix = reaction_matrix_left-reaction_matrix_right)')
group.add_argument(
    '--MConstraint',
    type=list or int,
    default=1,#default=1e-39,  
    help='Add different number constraint for each species. If the upper limit for all species is the same as --upper_limit, then set --Mconstraint=1')
group.add_argument(
    '--Conservation', 
    type=int,
    default=1, 
    help='imposing conservation of some quantities')
group.add_argument(
    '--training_step', type=int, default=1, help='Time step of iterating the dynamical equation P_tnew=T*P_t')
group.add_argument(
    '--deltaT', type=float, default=0.05, help='Time step length of iterating the dynamical equation')
group.add_argument(
    '--epoch1', type=int, default=10**4, help='maximum number of steps first few time steps')
group.add_argument(
    '--epoch2', type=int, default=50, help='maximum number of steps of later time steps')
group.add_argument(
    '--saving_data_time_step',
    type=list,
    default=[0,1e2,5e2,2e3,1e4,2e4,5e4,1e5,1.5e5,2e5,2.5e5,3e5,3.5e5,4e5,5e5,6e5,7e5,8e5,9e5,1e6],
    help='To save data at which time steps (give in a list)')
group.add_argument(
    '--training_loss_print_step',
    type=list,
    default=[0,1,2,101,1001,2e3,1e4,1e5,2e5,3e5,4e5,5e5],
    help='To print training loss at which time steps (give in a list)')

args = parser.parse_args()
