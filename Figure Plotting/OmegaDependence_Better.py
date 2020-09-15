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

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

n=30
N=1000
Omegalist=np.logspace(-2,0,num=n)
Lambdalist=np.logspace(-3,2,num=n)

gr = np.logspace(-2,2,num=N)



finaltwist = np.zeros((n,n))
delta0 = np.zeros((n,n))
r0 = np.zeros((n,n))

for j in range(n):
	for i in range(n):
		psi = np.loadtxt('Psi_'+str(i)+'_'+str(j)+'.csv')

		delta = np.loadtxt('Delta_'+str(i)+'_'+str(j)+'.csv')
		r0loc = np.where(delta==min(delta))[0][0]
		delta0[i,j] = min(delta)

		if np.where(psi == max(psi))[0][0]<len(psi)-1:
			finaltwist[i,j]=np.NaN
			r0[i,j]=np.NaN
		else:
			finaltwist[i,j] = psi[-1]
			r0[i,j] = gr[r0loc]






fig=plt.figure()
gs=gridspec.GridSpec(3,1,width_ratios=[1],height_ratios=[1,1,1])
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1])
ax3=plt.subplot(gs[2])

ax1.set_title('A)',loc='left',fontsize=20)
ax1.set_xlabel('$\Lambda$',fontsize=20)
ax1.set_ylabel('$\omega$',fontsize=20)
ax1.set_xscale('log')
ax1.set_yscale('log')
#ax1.title('$\\tilde{R}_0$ ($K=1$)',fontsize=20)

CS = ax1.contour(Lambdalist,Omegalist,r0,[0.1,0.2,0.5,1,2,5,10],colors='k')
ax1.clabel(CS, inline=1, fontsize=12)
ax1.minorticks_on()
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)

ax2.set_title('B)',loc='left',fontsize=20)
ax2.set_xlabel('$\Lambda$',fontsize=20)
ax2.set_ylabel('$\omega$',fontsize=20)
ax2.set_xscale('log')
ax2.set_yscale('log')
#ax2.title('Asymptotic $\psi(r)$ ($K=1$)',fontsize=20)

CS = ax2.contour(Lambdalist,Omegalist,finaltwist,colors='k')
ax2.clabel(CS, inline=1, fontsize=12)

ax2.minorticks_on()
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)


ax3.set_title('C)',loc='left',fontsize=20)
ax3.set_xlabel('$\Lambda$',fontsize=20)
ax3.set_ylabel('$\omega$',fontsize=20)
ax3.set_xscale('log')
ax3.set_yscale('log')

CS = ax3.contour(Lambdalist,Omegalist,delta0,[0.01,0.5,0.8,0.9,0.95,0.99],colors='k')
ax3.clabel(CS, inline=1, fontsize=12)
ax3.minorticks_on()
ax3.tick_params(axis='x', labelsize=14)
ax3.tick_params(axis='y', labelsize=14)


plt.tight_layout(pad=0.5)

plt.show()
