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

Omegalist=np.logspace(1,3,num=10)
Lambdalist=np.logspace(1,3,num=10)

gr = np.logspace(-2,3,num=100)

r0 = np.zeros((10,10))
finaltwist = np.zeros((10,10))

minloc = np.zeros((10,10))
minmag = np.zeros((10,10))
maxloc = np.zeros((10,10))
maxmag = np.zeros((10,10))

for i in range(10):
    for j in range(10):
        deltalist = np.loadtxt('Delta_0_3_'+str(j)+'_'+str(i)+'.csv')
        psi = np.loadtxt('Psi_0_3_'+str(j)+'_'+str(i)+'.csv')
        f = np.loadtxt('f_0_3_'+str(j)+'_'+str(i)+'.csv')
        
        d0 = min(deltalist)
        
        if np.where(psi == max(psi))[0][0]<99:
            r0[i,j] = np.NaN
            finaltwist[i,j]=np.NaN
        else:
            r0[i,j] = gr[np.where(deltalist==d0)[0][0]]
            finaltwist[i,j] = psi[-1]






fig=plt.figure()
gs=gridspec.GridSpec(2,1,width_ratios=[1],height_ratios=[1,1])
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1])

ax1.set_title('A)',loc='left',fontsize=20)
ax1.set_xlabel('$\log_{10} \Lambda$',fontsize=20)
ax1.set_ylabel('$\log_{10}\omega$',fontsize=20)
#ax1.title('$\\tilde{R}_0$ ($K=1$)',fontsize=20)

CS = ax1.contour(np.log10(Lambdalist),np.log10(Omegalist),r0,colors='k')
ax1.clabel(CS, inline=1, fontsize=10)
ax1.minorticks_on()
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)

ax2.set_title('B)',loc='left',fontsize=20)
ax2.set_xlabel('$\log_{10} \Lambda$',fontsize=20)
ax2.set_ylabel('$\log_{10}\omega$',fontsize=20)
#ax2.title('Asymptotic $\psi(r)$ ($K=1$)',fontsize=20)

CS = ax2.contour(np.log10(Lambdalist),np.log10(Omegalist),finaltwist,colors='k')
ax2.clabel(CS, inline=1, fontsize=12)

ax2.minorticks_on()
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
plt.tight_layout(pad=0.5)

plt.show()