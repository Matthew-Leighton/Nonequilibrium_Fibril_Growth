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


# This script needs access to the folder Lambda_Omega_ScanData

# Params used for the data:
n=60
N=600
Omegalist=np.logspace(-2,2,num=n)
Lambdalist=np.logspace(-3,2,num=n)
gr = np.logspace(-2,2,num=N)


# arrays to hold computed values of psi_infty, r_0 and delta_0
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

		if delta0[i,j]<0.1:
			r0[i,j]=np.NaN
			#finaltwist[i,j]=np.NaN

		finaltwist[i,j] = psi[-1]


# These functions are needed for the plotting function (specifically the curve in panel D)


def F(psi,gr,K,Lambda,delta,eta):
    return np.abs(gr - (1/2)*np.sin(2*psi) - K*np.tan(2*psi)*np.sin(psi)**2 - Lambda*((4*np.pi**2) - (eta**2) * np.cos(psi)**2)*np.tan(2*psi)*(delta*eta*gr)**2)

def psifunction(gr,K,deltalist,etalist,Lambda):
    psilist=np.zeros(len(gr))
    for i in range(len(gr)):
        psilist[i] = minimize(F,psilist[i-1],args=(gr[i],K,Lambda,deltalist[i],etalist[i])).x
    return psilist

def CalculateStructure_2(gr,K,Lambda,omega):
	N = len(gr)

	psi=np.zeros(N)
	etalist=np.zeros(N)
	deltalist=np.ones(N)

	for i in range(N):
		if i==0:
			psi[i] = gr[i]
			quadint2 = (gr[i])*(gr[i]*np.cos(psi[i])**2)/2
			quadint4 = (gr[i])*(gr[i]*np.cos(psi[i])**4)/2
		else:
			psi[i] = minimize(F,psi[i-1],args=(gr[i],K,Lambda,deltalist[i-1],etalist[i-1])).x
			quadint2 += (gr[i]-gr[i-1]) * (gr[i]*np.cos(psi[i])**2 + gr[i-1]*np.cos(psi[i-1])**2)/2
			quadint4 += (gr[i]-gr[i-1]) * (gr[i]*np.cos(psi[i])**4 + gr[i-1]*np.cos(psi[i-1])**4)/2
			etalist[i] = np.sqrt(4*(np.pi**2)*quadint2/quadint4)
			deltalist[i] =  ( 1 - (8*np.pi**4 * Lambda/omega) + (4*np.pi**2 * (Lambda/omega)*(1/(gr[i]**2))*etalist[i]**2 * quadint2) )
			deltalist[i] = np.sqrt(max(deltalist[i],0))

	return psi,etalist,deltalist




# The following code plots Figure 7

fig=plt.figure()
gs=gridspec.GridSpec(2,2,width_ratios=[1,1],height_ratios=[1,1])
ax1=plt.subplot(gs[0])
ax2=plt.subplot(gs[1])
ax3=plt.subplot(gs[2])
ax4 = plt.subplot(gs[3])

ax1.set_title('$R_0$',loc='right',fontsize=20)
ax1.set_title('A)',loc='left',fontsize=20)
ax1.set_xlabel('$\Lambda$',fontsize=20)
ax1.set_ylabel('$\omega$',fontsize=20)
ax1.set_xscale('log')
ax1.set_yscale('log')
#ax1.title('$\\tilde{R}_0$ ($K=1$)',fontsize=20)
ax1.contourf(Lambdalist,Omegalist,delta0,[0,0.1],alpha=1,colors='green')

CS = ax1.contour(Lambdalist,Omegalist,r0,[0.1,0.2,0.5,1,2],colors='k')
ax1.clabel(CS, inline=1, fontsize=12,manual=[(60,5),(5,5),(0.2,5),(0.03,5),(0.005,5)])

CS = ax1.contour(Lambdalist,Omegalist,delta0,[0.1],colors='green',linewidths=4)
#ax1.clabel(CS,fmt='', inline=1, fontsize=12)

ax1.minorticks_on()
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)

ax2.set_title('$\psi_\infty$',loc='right',fontsize=20)
ax2.set_title('B)',loc='left',fontsize=20)
ax2.set_xlabel('$\Lambda$',fontsize=20)
ax2.set_ylabel('$\omega$',fontsize=20)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.text(1,0.013,'$\psi_\infty =\pi/4$',fontsize=20)
#ax2.title('Asymptotic $\psi(r)$ ($K=1$)',fontsize=20)
ax2.contourf(Lambdalist,Omegalist,delta0,[0,0.1],alpha=1,colors='green')

#CS = ax2.contour(Lambdalist,Omegalist,delta0,[0.1],colors='green',linewidths=4)
#ax2.clabel(CS,fmt='', inline=1, fontsize=12)
CS = ax2.contour(Lambdalist,Omegalist,finaltwist,[0.1,0.2,0.3,0.4],colors='k')
ax2.clabel(CS, inline=1, fontsize=12,manual=[(7,5),(0.2,5),(0.03,5),(0.005,5)])

ax2.minorticks_on()
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)


ax3.set_title('$\delta_0$',loc='right',fontsize=20)
ax3.set_title('C)',loc='left',fontsize=20)
ax3.set_xlabel('$\Lambda$',fontsize=20)
ax3.set_ylabel('$\omega$',fontsize=20)
ax3.set_xscale('log')
ax3.set_yscale('log')

CS = ax3.contour(Lambdalist,Omegalist,delta0,[0,0.85,0.95,0.99,0.995,0.999],colors='k')
ax3.clabel(CS, inline=1, fontsize=12,manual=[(1,0.02),(10,0.07),(0.01,0.04),(5,0.4),(0.01,0.3),(0.01,2)])

#CS = ax3.contour(Lambdalist,Omegalist,delta0,[0],colors='green',linewidths=6)
#ax3.clabel(CS, inline=1, fontsize=12)
ax3.contourf(Lambdalist,Omegalist,delta0,[0,0.01],alpha=0.8,colors='green')

ax3.minorticks_on()
ax3.tick_params(axis='x', labelsize=14)
ax3.tick_params(axis='y', labelsize=14)


ax4.set_title('$\Lambda=10, \omega=0.02$',loc='right',fontsize=20)
ax4.set_title('D)',loc='left',fontsize=20)
ax4.set_xlabel('$r$',fontsize=20)
ax4.set_ylabel('$\psi$',fontsize=20)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlim(0.01,10000)
ax4.set_ylim(0.01,1.05)
psi = np.loadtxt('Psi_'+str(5)+'_'+str(48)+'.csv')
ax4.hlines(np.pi/4,0.05,10000,linestyle=':',color='xkcd:orange')
ax4.text(0.012,np.pi/4,'$\pi/4$',ha='left', va='center',color='xkcd:orange',fontsize=14)

gr = np.logspace(-2,4,num=1000)
psi,etalist,deltalist = CalculateStructure_2(gr,10,10,0.02)

ax4.plot(gr,psi,lw=3)

ax4.minorticks_on()
ax4.tick_params(axis='x', labelsize=14)
ax4.tick_params(axis='y', labelsize=14)


plt.tight_layout(pad=0.5)


plt.show()
