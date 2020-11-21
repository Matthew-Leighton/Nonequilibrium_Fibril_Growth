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


#This function calculates the free energy f(r) for a given structure and set of parameters

def CalculateFreeEnergy(gr,K,Lambda,omega,gamma,psi,deltalist,etalist):
	N=len(gr)

	fgamma = np.zeros(N)
	fK = np.zeros(N)
	fomega = np.zeros(N)
	fLambda = np.zeros(N)
	f = np.zeros(N)

	for i in range(N):
		if i==0:
			quadint2 = (gr[i])*(gr[i]*np.cos(psi[i])**2)/2
		else:
			quadint2 += (gr[i]-gr[i-1]) * (gr[i]*np.cos(psi[i])**2 + gr[i-1]*np.cos(psi[i-1])**2)/2

		fgamma[i] = gamma/gr[i]
		fK[i] = (1/2)*(np.sin(2*psi[i])/(2*gr[i]))**2 -(np.sin(2*psi[i])/(2*gr[i])) + (1/2)*K*((np.sin(psi[i])**4)/(gr[i]**2))

		if i==0:
			fomega[i] = omega * ( deltalist[i]**2 *(deltalist[i]**2 /2 -1)  )
		else:
			fomega[i] += omega * ( deltalist[i]**2 *(deltalist[i]**2 /2 -1)  + gr[i]*deltalist[i]*(deltalist[i]**2 -1)*(deltalist[i]-deltalist[i-1])/(gr[i]-gr[i-1]))
			fLambda[i] += Lambda * ( (1/2)*deltalist[i]**2 * (4*np.pi**2 - etalist[i]**2 * np.cos(psi[i])**2)**2  + 4*np.pi**2 * deltalist[i] * ((deltalist[i]-deltalist[i-1])/(gr[i]-gr[i-1]))*(2*np.pi**2 * gr[i] - etalist[i]**2 *(1/gr[i]) * quadint2)  )

		f[i] = fK[i] + fgamma[i] + fomega[i] + fLambda[i]

	return f


# Needs access to the folder K_Lambda_ParamScanData

##### Params used for the data:
n=40
omega=0.1
gamma=0.0
Klist=np.logspace(0,3,num=n)
Lambdalist=np.logspace(-3,2,num=n)


# Arrays for storing computed values of r0 and psi_infty
finaltwist = np.zeros((n,n))
r0 = np.zeros((n,n))
f_r0 = np.zeros((n,n))


for j in range(n):
    for i in range(n):
        deltalist = np.loadtxt('Delta_'+str(j)+'_'+str(i)+'.csv')
        psi = np.loadtxt('Psi_'+str(j)+'_'+str(i)+'.csv')
        etalist = np.loadtxt('Eta_'+str(j)+'_'+str(i)+'.csv')
        
        gr = np.logspace(-2,3,num=len(psi))
        f = CalculateFreeEnergy(gr,Klist[j],Lambdalist[i],0.1,gamma,psi,deltalist,etalist)
        
        d0 = min(deltalist)
        r0loc = np.where(deltalist==d0)[0][0]
        
        if np.where(psi == max(psi))[0][0]<len(psi)-1:
            r0[i,j] = np.NaN
            f_r0[i,j] = np.NaN
            finaltwist[i,j]=np.NaN
        else:
            r0[i,j] = gr[r0loc]
            f_r0[i,j] = f[r0loc]
            finaltwist[i,j] = psi[-1]

# Compute gamma_0:
gamma_0 = -f_r0*r0


# This function plots Figure 9:

def PlotGammaFig():

	plt.ylim(0.001,100)
	plt.xlim(1,1000)
	plt.xscale('log')
	plt.yscale('log') 
	CS = plt.contour(Klist,Lambdalist,gamma_0,[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10],colors='k')
	plt.clabel(CS, inline=1, fontsize=12)

	CS = plt.contour(Klist,Lambdalist,finaltwist,[0.09],colors='blue',linewidths=7,alpha=0.5)
	#plt.clabel(CS,fmt='Tendon (0.09)', inline=1, fontsize=16)

	plt.contourf(Klist,Lambdalist,finaltwist,[0.31,10],alpha=0.5,colors='orange')
	plt.contourf(Klist,Lambdalist,finaltwist,[0.01,0.09],alpha=0.5,colors='lightblue')
	plt.contourf(Klist,Lambdalist,finaltwist,[0.09,0.31],alpha=0.1,colors='grey')
	CS = plt.contour(Klist,Lambdalist,finaltwist,[0.31],colors='xkcd:orange',linewidths=7)
	#plt.clabel(CS,fmt='Cornea (0.31)', inline=1, fontsize=16)
	#plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.01,0.08],alpha=0.1,colors='grey')
	
	plt.title('$\gamma_0$',loc='right',fontsize=20)
	plt.xlabel('$K$',fontsize=20)
	plt.ylabel('$\Lambda$',fontsize=20)
	plt.minorticks_on()
	plt.tick_params(axis='x', labelsize=16)
	plt.tick_params(axis='y', labelsize=16)
	plt.tight_layout(pad=0.5)

	plt.show()

PlotGammaFig()




