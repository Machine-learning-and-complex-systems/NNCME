# NNCME
A software package for the manuscript "Neural-network solutions to stochastic reaction networks" (https://arxiv.org/abs/2210.01169)

(c) 2022, Lab of Machine Learning and Complex Systems
All rights reserved. 
We provide a code repository, which only requires to input the stoichiometric matrix, initial conditions, and reaction rates, allowing the approach to be applicable and adaptable to more stochastic reaction networks.

A detailed example on the methods is given in the main text. 

A guideline for the package is on the website: the website link will be generated after the manuscript is accepted. 

Contact: Ying Tang, jamestang23@gmail.com

--------------------------------------------------------------------------------------------------------------------------------------------

Guideline

A step-by-step guideline is on the website of the package: https://sites.google.com/view/dmipackage. The website link may differ after the manuscript is accepted.

--------------------------------------------------------------------------------------------------------------------------------------------

System requirements: 
All simulations were done using MATLAB® version R2019a.
We have used the toolbox “Hidden Markov Models (HMM)” in MATLAB.

Third-party packages: 

 (1) We have used the package of NFkB signaling model on https://github.com/biomystery/tnf_ikbd_nfkb_model.git.
 
 (2) The package to generate the data of NFkB on https://github.com/Adewunmi91/MACKtrack.
 
 (3) We thank Roy Wollman's group for sharing the code of vector method.
 
 (4) The decoding-based method was not included here, because it can be separately implemented by the user-friendly package (https://github.com/swainlab/mi-by-decoding).

Expected run time: all the expected run time below is evaluated based on a personal desktop with intel(R) core(tm) i7-8700 CPU @ 3.7GHz.
