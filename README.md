# NNCME
(c) Lab of Machine Learning and Complex Systems (2022).
All rights reserved. 

A software package for the manuscript "Neural-network solutions to stochastic reaction networks" (https://arxiv.org/abs/2210.01169)

NNCME stands for Neural Network Chemical Master Equation. It tracks the time evolution of stochastic reaction networks by training the variational autoregressive neural network, which generates the time evolution of the joint probability distribution and the marginal statistics. This approach is applicable and adaptable to general stochastic reaction networks. 


--------------------------------------------------------------------------------------------------------------------------------------------

System requirements: 
All simulations were done using Python.
We have used the package Pytorch. The code requires Python >= 3.6 and PyTorch >= 1.0.

--------------------------------------------------------------------------------------------------------------------------------------------

# Inputs

For a new model, the users can conveniently input their system as a `.py` file: please refer to the existing example. The input arguments include:

(1) the stoichiometric matrix, 

(2) reaction rates, 

(3) propensities,

(4) initial conditions, 

(5) hyperparameters of the neural network.

To create a new `.py` file for your model, typically you only need to input the above (1) (2) (3) (4) in the function 'rates'. Then, please run MasterEq.py (see below).

Note: 

(a) The stoichiometric matrix needs to be put separately for the left and right side of the reactions, in the variable "ReactionMatLeft" and "ReactionMatRight" separately. The stoichiometric matrix = ReactionMatRight - ReactionMatLeft.

(b) The reaction rate variable is "r". For time-dependent rates, please refer to the example Epidemic.

(c) The default propensities follow the law of mass action. Other types such as the Hill function can be implemented: refer to the example cascade2.

(d) The default initial condition is the discrete delta distribution concentrated on the given number in the variable "initialD". Other types of distribution such as Poisson distribution can be added to the code if the users have the demand.

(e) For hyperparameters such as dt, please use those in the Supplementary table II,III of the manuscript as a reference for your model.

(f) Optional: Both the recurrent neural network (RNN) and the transformer can be chosen as the unit of the neural-network model, as an option args.net='rnn' or 'transformer'.

(g) Optional: If certain species count has a constrained maximum limit, please put it into the variable "MConstrain"

(h) Optional: For some examples (e.g. toggle switch), there is another function with name 'Mask' in its `.py` file. The 'Mask' function allows only the reactions with such a proper count to occur. For example, DNA typically has the count of 0 or 1 inside a cell.

(i) Optional: If the total species count obeys a conservation law, please set the variable "conservation=initialD.sum()" as the total initial count of all species.

--------------------------------------------------------------------------------------------------------------------------------------------

# Platforms

To implement the code after providing the `.py` file, there are two ways:

(a) Server: you can use a `.sh` to input the hyperparameters from the shell. Please refer to scripts `.sh` in the folder "Shell".

(b) PC users (Windows): you may use Spyder to run MasterEq.py by calling the name of your model script. You can properly adjust the hyperparameters in MasterEq.py, including those listed below.

```
###Add models begin (necessary changes according to the new model)----------------------------------
from ModelName import ModelName  #from your ModelName.py file import the model class
###Add models end----------------------------------

###Set hyperparameters begin (necessary changes according to the new model)-------------------------------
args.Model='ModelName' #Change to your model name
args.L=4 #The number of species
args.M=int(80) #Upper limit on the count of species
args.batch_size=1000 #Number of batch samples
args.Tstep=100# Time step of iterating the chemical master equation
args.delta_t=0.0005 #Time step length of iterating the chemical master equation

#More hyperparameters of the neural network (unnecessary changes)
#Please refer to Supplementary table II,III of the manuscript for the choice of the above hyperparameters if changes are required. 
#If no specification, the parameters will be the default in MasterEq.py and args.py
###Set hyperparameters end-------------------------------

###Add model command begin (necessary changes according to the new model)----------------------------
if args.Model=='ModelName':
    model = ModelName(**vars(args))   
###Add model command end----------------------------
```
--------------------------------------------------------------------------------------------------------------------------------------------

# Examples

Examples of the methods are given in the main text. The representative examples include:  

(1) the genetic toggle switch, 

(2) the early life self-replicator, 

(3) the epidemic model, 

(4) the intracellular signaling cascade. 

They separately demonstrate that our approach is applicable to systems with a multimodal distribution, with an intrinsic constraint of count conservation, with time-dependent parameters, and in high dimensions.

Scripts `.sh` in the folder "Shell' are commands to reproduce the results in Fig. 2~5. Directly running these scripts several GPU hours. Expected run time for the examples are provided in the Supplementary table II,III of the manuscript: All computations are performed with a single core GPU (~25% usage) of a Tesla-V100. In practice, one may run these commands with different hyperparameters in parallel on multiple GPUs.

--------------------------------------------------------------------------------------------------------------------------------------------


A step-by-step guideline of implementing an example is on the Jupyter notebook: `A Simple Template.ipynb` and `Detailed Version of Gene Expression.ipynb`. 

Contact: Ying Tang, jamestang23@gmail.com; Jiayu Weng, 202011059131@mail.bnu.edu.cn

