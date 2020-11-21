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



# Needs access to the folder K_Lambda_ParamScanData

##### params used for data:
omega=0.1
gamma=0.0
n=40
Klist=np.logspace(0,3,num=n)
Lambdalist=np.logspace(-3,2,num=n)


#array for storing computed values of r_0 and psi_infty
finaltwist = np.zeros((n,n))
r0 = np.zeros((n,n))



#Computes r_0
for j in range(n):
    for i in range(n):
        deltalist = np.loadtxt('Delta_'+str(j)+'_'+str(i)+'.csv')
        psi = np.loadtxt('Psi_'+str(j)+'_'+str(i)+'.csv')
        
        gr = np.logspace(-2,3,num=len(psi))
        
        d0 = min(deltalist)
        r0loc = np.where(deltalist==d0)[0][0]
        
        if np.where(psi == max(psi))[0][0]<len(psi)-1:
            r0[i,j] = np.NaN
            finaltwist[i,j] = np.NaN
        else:
            r0[i,j] = gr[r0loc]
            finaltwist[i,j] = psi[-1]


# this function plots Figure 8

def PlotR0():

	plt.ylim(0.001,100)
	plt.xlim(1,1000)
	plt.xscale('log')
	plt.yscale('log') 
	CS = plt.contour(Klist,Lambdalist,r0,[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50],colors='k')
	plt.clabel(CS, inline=1, fontsize=12)

	CS = plt.contour(Klist,Lambdalist,finaltwist,[0.09],colors='blue',linewidths=7,alpha=0.5)
	#plt.clabel(CS,fmt='Tendon (0.09)', inline=1, fontsize=16)

	plt.contourf(Klist,Lambdalist,finaltwist,[0.31,10],alpha=0.5,colors='orange')
	plt.contourf(Klist,Lambdalist,finaltwist,[0.01,0.09],alpha=0.5,colors='lightblue')
	plt.contourf(Klist,Lambdalist,finaltwist,[0.09,0.31],alpha=0.1,colors='grey')
	CS = plt.contour(Klist,Lambdalist,finaltwist,[0.31],colors='xkcd:orange',linewidths=7)
	#plt.clabel(CS,fmt='Cornea (0.31)', inline=1, fontsize=16)
	#plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.01,0.08],alpha=0.1,colors='grey')
	
	plt.title('$R_0$',loc='right',fontsize=20)
	plt.xlabel('$K$',fontsize=20)
	plt.ylabel('$\Lambda$',fontsize=20)
	plt.minorticks_on()
	plt.tick_params(axis='x', labelsize=16)
	plt.tick_params(axis='y', labelsize=16)
	plt.tight_layout(pad=0.5)

	plt.show()


PlotR0()



