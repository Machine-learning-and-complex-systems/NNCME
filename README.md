# NNCME
A software package for the manuscript "Neural-network solutions to stochastic reaction networks" (https://arxiv.org/abs/2210.01169)
NNCME stands for Neural Network Chemical Master Equation.

(c) 2022, Lab of Machine Learning and Complex Systems
All rights reserved. 

A code package based on pytorch in Python with a documentation is provided. This approach is applicable and adaptable to general stochastic reaction networks.

--------------------------------------------------------------------------------------------------------------------------------------------

# Inputs

The input arguments include 
(1) the stoichiometric  matrix, 
(2) initial conditions, 
(3) reaction rates, 
(4) propensities and hyperparameters. 

Then, the users only need to use a function to train the VAN, and to generate the time evolution of the joint probability distribution and the marginal statistics.


--------------------------------------------------------------------------------------------------------------------------------------------

# Examples

Examples on the methods are given in the main text. The representative examples include  

(1) the genetic toggle switch, 

(2) the early life self-replicator, 

(3) the epidemic model, 

(4) the intracellular signaling cascade. 

They separately demonstrate that our approach is applicable to systems with a multimodal distribution, with an intrinsic constraint of count conservation, with time-dependent parameters, and in high dimensions.

Contact: Ying Tang, jamestang23@gmail.com, 202011059131@mail.bnu.edu.cn

--------------------------------------------------------------------------------------------------------------------------------------------

Guideline

A step-by-step guideline is on the website of the package: the website link will be generated after the manuscript is accepted. 

--------------------------------------------------------------------------------------------------------------------------------------------

System requirements: 
All simulations were done using Python.
We have used the Pytorch.

Expected run time for the examples are provided in the Supplementary table of the manuscript: All computational are performed with a single core GPU (~25% usage) of a Tesla-V100.
