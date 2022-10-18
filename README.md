# NNCME
(c) Lab of Machine Learning and Complex Systems (2022).
All rights reserved. 

A software package for the manuscript "Neural-network solutions to stochastic reaction networks" (https://arxiv.org/abs/2210.01169)

NNCME stands for Neural Network Chemical Master Equation. This approach is applicable and adaptable to general stochastic reaction networks. The software package is in Python. 


--------------------------------------------------------------------------------------------------------------------------------------------

System requirements: 
All simulations were done using Python.
We have used the package Pytorch. The code requires Python >= 3.6 and PyTorch >= 1.0.

--------------------------------------------------------------------------------------------------------------------------------------------

# Inputs

The users can conveniently input their system as a .py file: please refer to the existing example. The input arguments include:

(1) the stoichiometric matrix, 

(2) reaction rates, 

(3) propensities,

(4) initial conditions, 

(5) hyperparameters of the neural network.

For hyperparameters, please use those in the Supplementary table II,III of the manuscript as a reference for your example. Then, you only need to use a function to train the VAN, and to generate the time evolution of the joint probability distribution and the marginal statistics. Both the recurrent neural network (RNN) and the transformer can be chosen as the unit of the neural-network model, as an option in this package.

When exploring more models, after adding a .py file of the system please ensure to add more details and change the parameters in MasterEq.py. Necessary changes in the code are listed below.

```
###Add models----------------------------------
from ModelName import ModelName  #from your ModelName.py file import the model class

##Set parameters-------------------------------
###Initialize parameters: otherwise the parameters are specified in init_out_dir-> args.py
args.Model='ModelName' #Change to your model name
args.L=15 #Species number
args.M=int(80) #Upper limit of the molecule number
#...........More parameters in the code

###Add model command----------------------------
if args.Model=='ModelName':
    model = ModelName(**vars(args))   

```
--------------------------------------------------------------------------------------------------------------------------------------------

# Examples

Examples on the methods are given in the main text. The representative examples include:  

(1) the genetic toggle switch, 

(2) the early life self-replicator, 

(3) the epidemic model, 

(4) the intracellular signaling cascade. 

They separately demonstrate that our approach is applicable to systems with a multimodal distribution, with an intrinsic constraint of count conservation, with time-dependent parameters, and in high dimensions.

Scripts `xxx.sh` are commands to reproduce the results in Fig. 2~5. Directly running these scripts several GPU hours. Expected run time for the examples are provided in the Supplementary table II,III of the manuscript: All computational are performed with a single core GPU (~25% usage) of a Tesla-V100. In practice, one may run these commands with different hyperparameters in parallel on multiple GPUs.

--------------------------------------------------------------------------------------------------------------------------------------------


A step-by-step guideline is on the website: the link will be generated after the manuscript is accepted. 

Contact: Ying Tang, jamestang23@gmail.com; Jiayu Weng, 202011059131@mail.bnu.edu.cn

