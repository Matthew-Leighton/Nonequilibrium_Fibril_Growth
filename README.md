# Nonequilibrium Fibril Growth

This repository contains all the electronic material (code, data, etc) needed to replicate the findings of the manuscript titled "Nonequilibrium Growth and Twist of Cross-Linked Collagen Fibrils" by Matthew P Leighton, Laurent Kreplak, and Andrew D Rutenberg. In particular, the contents include:

## StructureCalculations.py
This file contains a number of defined functions, which were used to generate the various datasets visualized in the manuscript. These functions solve for fibril structure ( psi(r), eta(r), delta(r), f(r) ) for a given set of parameters.

## FinalFigures
This folder contains pdf and tiff copies of all the figures that are in the manuscript.

## FigureCode
This folder contains scripts for producing each of the figures in the manuscript. Note that each script must be run in a specific data subfolder, which is indicated by comments in the script.

## Arxiv Submission
The latex code, figures, and .bib file that comprise the version of the manuscript submitted to the arxiv.

## Data
This folder holds all the different data sets needed to recreate the figures in the manuscript.

Data sets are as follows:
- K_Lambda_ParamScanData
	* psi, eta, delta, and f function of r for fixed gamma=0.01,omega=0.1, and varying K and Lambda. The r array that corresponds to the data is r = np.logspace(-2,3,num=1000). K is varied through np.logspace(1,3,num=40) while Lambda is varied through np.logspace(-3,2,num=40).
	* This data was used for Figures 3,8, and 9.
- TendonData
	* psi, eta, delta, and f functions of r for K=100, Lambda=0.5, omega=0.1, and gamma=0.01. Again, the corresponding r is r = np.logspace(-2,3,num=1000).
	* This data was used for Figures 2,4, and 5.
- Lambda_Omega_ScanData
	* psi, eta, delta, and f functions of r for K=10, gamma=0.01, and varying Lambda and omega. The corresponding r is r = np.logspace(-2,2,num=600), while omega and Lambda are both in np.logspace(-2,2,num=60).
	* This data was used for Figure 7.
- SmallK_SmallLambda_Data
	* psi, eta, delta, and f functions of r for omega=0.1, gamma=0.01, and varying K and Lambda. The r array that corresponds to the data is r = np.logspace(-2,3,num=1000). K is varied through np.logspace(1,2.5,num=40) while Lambda is varied through np.logspace(-3,-1,num=40).
	* This data was used for Figures 6 and 10
