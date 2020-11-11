import math as m
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar,minimize
import scipy.integrate as integrate
import matplotlib.gridspec as gridspec
from scipy.misc import derivative
from scipy.optimize import fsolve
import matplotlib.cm as cmap
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from scipy import ndimage

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


##### Params:
K=100
Lambda=0.5
omega=0.1
gamma=0.01
gr = np.logspace(-2,3,num=1000)

# Needs access to Param Scan Data





def PlotDelta0():
	#Needs to run in KLambdaParamScanData
	N=1000
	n=40

	Klist=np.logspace(0,3,num=n)
	Lambdalist=np.logspace(-3,2,num=n)
	delta0 = np.zeros((n,n))

	for j in range(n):
	    for i in range(n):
	        delta = np.loadtxt('Delta_'+str(j)+'_'+str(i)+'.csv')
	        
	        delta0[i,j] = min(delta)


	plt.ylim(0.001,100)
	plt.xlim(1,1000)
	plt.xscale('log')
	plt.yscale('log') 
	CS = plt.contour(Klist,Lambdalist,delta0,[0.9,0.91,0.92,0.94,0.96,0.98,0.99,0.995,0.999,0.9995,0.9999],colors='k')
	plt.clabel(CS, inline=1, fontsize=12)
	'''
	CS = plt.contour(Klist,Lambdalist,finaltwist,[0.09],colors='blue',linewidths=10,alpha=0.5)
	plt.clabel(CS,fmt='Tendon (0.09)', inline=1, fontsize=16)

	plt.contourf(Klist,Lambdalist,finaltwist,[0.31,1],alpha=0.5,colors='orange')
	#plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.08,0.1],alpha=0.5,colors='blue')
	#plt.text(0.35,-1.8,'Cornea',color='black',fontsize=16)
	plt.contourf(Klist,Lambdalist,finaltwist,[0.01,0.31],alpha=0.1,colors='grey')
	CS = plt.contour(Klist,Lambdalist,finaltwist,[0.31],colors='xkcd:orange',linewidths=10)
	plt.clabel(CS,fmt='Cornea (0.31)', inline=1, fontsize=16)
	#plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.01,0.08],alpha=0.1,colors='grey')
	'''
	plt.xlabel('$K$',fontsize=20)
	plt.ylabel('$\Lambda$',fontsize=20)
	plt.minorticks_on()
	plt.tick_params(axis='x', labelsize=14)
	plt.tick_params(axis='y', labelsize=14)
	plt.tight_layout(pad=0.5)

	plt.show()


PlotDelta0()



