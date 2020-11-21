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


# This script needs access to the folder SmallK_SmallLambda_Data

##### Params used for the data:
omega=0.1
gamma=0.01
gr = np.logspace(-2,3,num=1000)
n=20
Klist=np.logspace(0,2.5,num=n)
Lambdalist=np.logspace(-3,-1,num=n)



def CalculateFprime(gr,K,Lambda,omega,gamma,psi,deltalist,etalist):
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
        
    fprime = np.zeros(N)
    for i in range(N):
        if i==0:
            fprime[i] = 0
        else:
            fprime[i] = (f[i]-f[i-1])/(gr[i]-gr[i-1])

    return fprime


##### Data analysis, computation of relevant quantities

r0 = np.zeros((n,n))
finaltwist = np.zeros((n,n))
r0twist = np.zeros((n,n))
R_C = np.zeros((n,n))
fprime_rc = np.zeros((n,n))


for j in range(n):
    for i in range(n):
        deltalist = np.loadtxt('Delta_'+str(j)+'_'+str(i)+'.csv')
        etalist = np.loadtxt('Eta_'+str(j)+'_'+str(i)+'.csv')
        psi = np.loadtxt('Psi_'+str(j)+'_'+str(i)+'.csv')
        
        gr = np.logspace(-2,3,num=len(psi))
        fprime = CalculateFprime(gr,Klist[j],Lambdalist[i],0.1,gamma,psi,deltalist,etalist)
        
        d0 = min(deltalist)
        r0loc = np.where(deltalist==d0)[0][0]
        
        biggerthancornea = np.where(psi>=0.31)[0]
        if len(biggerthancornea)>0:
            R_C[i,j] = gr[min(biggerthancornea)]
            fprime_rc[i,j] = fprime[min(biggerthancornea)]
        else:
            R_C[i,j] = np.NaN
            fprime_rc[i,j] = np.NaN
        
        if np.where(psi == max(psi))[0][0]<len(psi)-1:
            r0[i,j] = np.NaN
            finaltwist[i,j]=np.NaN
        else:
            r0[i,j] = gr[r0loc]
            finaltwist[i,j] = psi[-1]



##### Computing concentration control requirements

n=20

k_24_lower = np.zeros((n,n))
k_24_upper = np.zeros((n,n))

tilde_R_C = 15*10**(-9) # m
qlower = 1/(10**(-6)) # /m
qupper = 5*np.pi*qlower # /m

kBT = 4.3 * (10**(-21)) # Joules
nlower = 2*10**17 # /m^3
nupper = 10**20 # /m^3

K22_lower = 0.6*10**(-12) #N
K22_upper = 6*10**(-12) #N

nf = 1.32*10**24 # /m^3

k_24_lower = 1 - (qupper*tilde_R_C/R_C)
k_24_upper = 1 - (qlower*tilde_R_C/R_C)

beta_inverse_lower = nlower*kBT*(1-k_24_upper) / ( K22_upper * (qupper**2) * (1+k_24_upper) )
beta_inverse_upper = nupper*kBT*(1-k_24_lower) / ( K22_lower * (qlower**2) * (1+k_24_lower) )

deltar_r_lower = beta_inverse_lower / (fprime_rc *R_C)
deltar_r_upper = beta_inverse_upper / (fprime_rc *R_C)

maxdeltanovern = 0.16 * R_C * fprime_rc * K22_upper * (qupper**2) * (1+k_24_lower) / ( (nf * kBT) * (1-k_24_lower) )


# This function plots Figure 10

def Fig10():
	fig=plt.figure()
	gs=gridspec.GridSpec(2,1,width_ratios=[1],height_ratios=[1,1])

	ax1=plt.subplot(gs[0])
	ax2=plt.subplot(gs[1])

	#ax1.imshow(k_24_lower,cmap=cmap.autumn,origin='lower',extent=[0,3,-3,2],aspect=3/5)#,interpolation='gaussian')
	ax1.set_xlabel('$ K$',fontsize=20)
	ax1.set_ylabel('$\Lambda$',fontsize=20)
	ax1.set_title("A)",loc='left',fontsize=20)
	ax1.set_title('$\Delta n/n$',loc='right',fontsize=20)
	ax1.set_xscale('log')
	ax1.set_yscale('log')

	CS = ax1.contour(Klist,Lambdalist,maxdeltanovern,[0.04,0.05,0.06,0.07,0.08],colors='k')#,[-1.5,-1,-0.5,-0.2,-0.1,-0.01],colors='k')
	ax1.clabel(CS, inline=1, fontsize=14)

	CS = ax1.contour(Klist,Lambdalist,finaltwist,[0.31],colors='xkcd:orange',linewidths=3)
	ax1.text(23,0.007,'$\psi_\infty \leq 0.31$',fontsize=20,rotation = -22.5)
	ax1.contourf(Klist,Lambdalist,finaltwist,[0.31,0.7],alpha=0.5,colors='orange')


	ax1.minorticks_on()
	ax1.tick_params(axis='x', labelsize=16)
	ax1.tick_params(axis='y', labelsize=16)

	#ax2.imshow(k_24_upper,cmap=cmap.autumn,origin='lower',extent=[0,3,-3,2],aspect=3/5)#,interpolation='gaussian')
	ax2.set_xlabel('$ K$',fontsize=20)
	ax2.set_ylabel('$\Lambda$',fontsize=20)
	ax2.set_title("B)",loc='left',fontsize=20)
	ax2.set_title('$R_C/R_0$',loc='right',fontsize=20)
	ax2.set_xscale('log')
	ax2.set_yscale('log')

	CS = ax2.contour(Klist,Lambdalist,R_C/r0,[0.1,0.2,0.3,0.5,1,2],colors='k')
	ax2.clabel(CS, inline=1, fontsize=14)

	CS = ax2.contour(Klist,Lambdalist,finaltwist,[0.31],colors='xkcd:orange',linewidths=3)
	#ax2.text(40,0.0009,'$\psi_\infty \leq 0.31$',fontsize=20,rotation = -19)
	ax2.contourf(Klist,Lambdalist,finaltwist,[0.31,0.7],alpha=0.5,colors='orange')

	ax2.minorticks_on()
	ax2.tick_params(axis='x', labelsize=16)
	ax2.tick_params(axis='y', labelsize=16)
	plt.tight_layout(pad=0.5)

	plt.show()

Fig10()


