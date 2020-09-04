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
K=500
Lambda=0.5
omega=0.1
gamma=0.01
gr = np.logspace(-2,3,num=1000)

# Needs access to Param Scan Data


##### Plot Asymptotic Twist Figure:
def PlotPsiInftyMap():
	#Needs to run in KLambdaParamScanData
	N=1000
	n=10

	Klist=np.logspace(0,3,num=n)
	Lambdalist=np.logspace(-3,2,num=n)
	finaltwist = np.zeros((n,n))

	for j in range(n):
	    for i in range(n):
	        psi = np.loadtxt('Psi_'+str(j)+'_'+str(i)+'.csv')
	        
	        gr = np.logspace(-2,3,num=len(psi))
	        
	        if np.where(psi == max(psi))[0][0]<len(psi)-1:
	            finaltwist[i,j]=np.NaN
	        else:
	            finaltwist[i,j] = psi[-1]
	        
	finaltwist[2,0] = 0.42

	#finaltwist = ndimage.filters.gaussian_filter(finaltwist,0.01)

	#plt.imshow(finaltwist,cmap=cmap.autumn,origin='lower',extent=[0,3,-3,2],aspect=3/5,interpolation='gaussian')
	plt.xlabel('$\log_{10} K$',fontsize=20)
	plt.ylabel('$\log_{10} \Lambda$',fontsize=20)
	
	CS = plt.contour(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.06,0.16,0.24,0.4,0.48,0.56],colors='k')
	plt.clabel(CS, inline=1, fontsize=12)

	CS = plt.contour(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.09],colors='blue',linewidths=15,alpha=0.5)
	plt.clabel(CS,fmt='Tendon (0.09)', inline=1, fontsize=16)

	plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.31,0.7],alpha=0.2,colors='orange')
	#plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.08,0.1],alpha=0.5,colors='blue')
	#plt.text(0.35,-1.8,'Cornea',color='black',fontsize=16)
	plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.01,0.31],alpha=0.1,colors='grey')

	CS = plt.contour(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.31],colors='xkcd:orange')
	plt.clabel(CS,fmt='Cornea (0.31)', inline=1, fontsize=16)
	#plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.01,0.08],alpha=0.1,colors='grey')
	'''

	plt.ylim(0.001,100)
	plt.xlim(1,1000)
	plt.xscale('log')
	plt.yscale('log') 
	CS = plt.contour(Klist,Lambdalist,finaltwist,[0.06,0.16,0.24,0.31,0.4,0.48,0.56],colors='k')
	plt.clabel(CS, inline=1, fontsize=12)

	CS = plt.contour(Klist,Lambdalist,finaltwist,[0.09],colors='blue',linewidths=15,alpha=0.5)
	plt.clabel(CS,fmt='Tendon', inline=1, fontsize=16)

	plt.contourf(Klist,Lambdalist,finaltwist,[0.31,0.7],alpha=0.5,colors='orange')
	#plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.08,0.1],alpha=0.5,colors='blue')
	plt.text(0.35,-1.8,'Cornea',color='black',fontsize=16)
	plt.contourf(Klist,Lambdalist,finaltwist,[0.01,0.31],alpha=0.1,colors='grey')
	#plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.01,0.08],alpha=0.1,colors='grey')
	'''

	plt.minorticks_on()
	plt.tick_params(axis='x', labelsize=14)
	plt.tick_params(axis='y', labelsize=14)
	plt.tight_layout(pad=0.5)

	plt.show()


PlotPsiInftyMap()




plt.ylim(0.001,100)
plt.xlim(1,1000)
plt.ylabel('$\Lambda$',fontsize=20)
plt.xlabel('$K$',fontsize=20)
plt.xscale('log')
plt.yscale('log')
plt.minorticks_on()
plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)
plt.tight_layout(pad=0.5)

plt.xscale('lin')
plt.yscale('lin')
CS = plt.contour(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.06,0.16,0.24,0.4,0.48,0.56],colors='k')
plt.clabel(CS, inline=1, fontsize=12)

CS = plt.contour(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.09],colors='blue',linewidths=15,alpha=0.5)
plt.clabel(CS,fmt='Tendon (0.09)', inline=1, fontsize=16)

plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.31,0.7],alpha=0.2,colors='orange')
#plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.08,0.1],alpha=0.5,colors='blue')
#plt.text(0.35,-1.8,'Cornea',color='black',fontsize=16)
plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.01,0.31],alpha=0.1,colors='grey')

CS = plt.contour(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.31],colors='xkcd:orange')
plt.clabel(CS,fmt='Cornea (0.31)', inline=1, fontsize=16)



plt.show()

