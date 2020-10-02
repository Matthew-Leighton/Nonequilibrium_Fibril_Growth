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


##### Params:
K=500
Lambda=0.5
omega=0.1
gamma=0.01
gr = np.logspace(-2,3,num=1000)

# Needs access to Param Scan Data


#####Data Analysis Part 1
N=1000
n=20

Klist=np.logspace(0,2.5,num=n)
Lambdalist=np.logspace(-3,-1,num=n)

gamma = 0.01

r0 = np.zeros((n,n))
finaltwist = np.zeros((n,n))
r0twist = np.zeros((n,n))
fprime_max = np.zeros((n,n))
psi_rmax = np.zeros((n,n))
R_C = np.zeros((n,n))
fprime_r0 = np.zeros((n,n))
rmax= np.zeros((n,n))
fprime_rc = np.zeros((n,n))
fprimemax_restricted = np.zeros((n,n))
rmax_restricted = np.zeros((n,n))
psirmax_restricted = np.zeros((n,n))

minloc = np.zeros((n,n))
minmag = np.zeros((n,n))
maxloc = np.zeros((n,n))
maxmag = np.zeros((n,n))

test = np.zeros((n,n))

for j in range(n):
    for i in range(n):
        test[i,j] = np.sqrt(Klist[j]/Lambdalist[i] - 1/(3*Lambdalist[i]))/(4*np.pi**2)
        deltalist = np.loadtxt('Delta_'+str(j)+'_'+str(i)+'.csv')
        etalist = np.loadtxt('Eta_'+str(j)+'_'+str(i)+'.csv')
        psi = np.loadtxt('Psi_'+str(j)+'_'+str(i)+'.csv')
        
        gr = np.logspace(-2,3,num=len(psi))
        fprime = CalculateFprime(gr,Klist[j],Lambdalist[i],0.1,gamma,psi,deltalist,etalist)
        f = CalculateFreeEnergy(gr,Klist[j],Lambdalist[i],0.1,gamma,psi,deltalist,etalist)
        
        d0 = min(deltalist)
        r0loc = np.where(deltalist==d0)[0][0]
        rmaxloc = np.where(fprime==max(fprime))[0][0]
        
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
            r0twist[i,j]=np.NaN
            fprime_max[i,j] = np.NaN
            psi_rmax[i,j]=np.NaN
            fprime_r0[i,j] = np.NaN
            rmax[i,j] = np.NaN
        else:
            r0[i,j] = gr[r0loc]
            finaltwist[i,j] = psi[-1]
            r0twist[i,j] = psi[r0loc]
            fprime_max[i,j] = max(fprime)
            psi_rmax[i,j] = psi[rmaxloc]
            fprime_r0[i,j] = fprime[r0loc]
            rmax[i,j] = gr[rmaxloc]

    #finaltwist[2,0] = 0.42
    #r0[2,0] = 0.8


#####Data Analysis Part 2

#fprime_rc = ndimage.filters.gaussian_filter(fprime_rc,1)
#R_C = ndimage.filters.gaussian_filter(R_C,1)
n=20

k_24_lower = np.zeros((n,n))
k_24_upper = np.zeros((n,n))

tilde_R_C = 15*10**(-9) # m
qlower = 1/(10**(-6)) # /m
qupper = 12*qlower # /m

kBT = 4.3 * (10**(-21)) # Joules
nlower = 2*10**17 # /m^3
nupper = 10**20 # /m^3

K22_lower = 0.6*10**(-12) #N
K22_upper = 6*10**(-12) #N

nf = 1.32*10**24

k_24_lower = 1 - (qupper*tilde_R_C/R_C)
k_24_upper = 1 - (qlower*tilde_R_C/R_C)

beta_inverse_lower = nlower*kBT*(1-k_24_upper) / ( K22_upper * (qupper**2) * (1+k_24_upper) )
beta_inverse_upper = nupper*kBT*(1-k_24_lower) / ( K22_lower * (qlower**2) * (1+k_24_lower) )


deltar_r_lower = beta_inverse_lower / (fprime_rc *R_C)
deltar_r_upper = beta_inverse_upper / (fprime_rc *R_C)



#maxdeltanovern = 0.05 * R_C * fprime_rc * K22_upper * (qupper**2) * (1+k_24_upper) / ( (nf * kBT) * (1-k_24_upper) )
maxdeltanovern = 0.16 * R_C * fprime_rc * K22_upper * (qupper**2) * (1+k_24_upper) / ( (nf * kBT) * (1-k_24_upper) )

maxdeltanovern_justdeltar = (10/40)*R_C * fprime_rc * K22_upper * (qupper**2) * (1+k_24_upper) / ( (nf * kBT) * (1-k_24_upper) )



#####Data Analysis Part 3
'''
M=100
array2 = maxdeltanovern
for i in range(n):
	for j in range(n):
		if np.isnan(array2[i,j]):
			array2[i,j] = 0.3
array2 = ndimage.filters.gaussian_filter(array2,0.1)
bettercontroldata = ndimage.zoom(array2, M)
betterKdata = ndimage.zoom(Klist, M)
betterLambdadata = ndimage.zoom(Lambdalist, M)
blurredr0 = ndimage.filters.gaussian_filter(r0,0.2)'''




def PlotConcentrationControl():
	#plt.imshow(maxdeltanovern,cmap=cmap.autumn,origin='lower',extent=[1,10**2.5,0.001,0.1],aspect=0.8,interpolation='gaussian')
	plt.xlabel('$K$',fontsize=20)
	plt.ylabel('$\Lambda$',fontsize=20)
	plt.xscale('log')
	plt.yscale('log')
	#plt.title("max{$\Delta n/n$}",fontsize=20)

	CS = plt.contour(Klist,Lambdalist,finaltwist,[0.31],colors='xkcd:orange',linewidths=3)
	plt.text(30,0.006,'$\psi_\infty \leq 0.31$',fontsize=20,rotation = -38)
	plt.contourf(Klist,Lambdalist,finaltwist,[0.31,0.7],alpha=0.5,colors='orange')

	CS = plt.contour(Klist,Lambdalist,maxdeltanovern)#,[0.05,0.12,0.16,0.2,0.23],colors='k')
	plt.clabel(CS, inline=1, fontsize=20)

	plt.minorticks_on()
	plt.tick_params(axis='x', labelsize=14)
	plt.tick_params(axis='y', labelsize=14)
	plt.tight_layout(pad=0.5)

	plt.show()



# Plot k24 Bounds:
def K24Bounds():
	fig=plt.figure()
	gs=gridspec.GridSpec(2,1,width_ratios=[1],height_ratios=[1,1])

	ax1=plt.subplot(gs[0])
	ax2=plt.subplot(gs[1])

	#ax1.imshow(k_24_lower,cmap=cmap.autumn,origin='lower',extent=[0,3,-3,2],aspect=3/5)#,interpolation='gaussian')
	ax1.set_xlabel('$ K$',fontsize=20)
	ax1.set_ylabel('$\Lambda$',fontsize=20)
	ax1.set_title("$k_{24}^{min}$",fontsize=20)
	ax1.set_xscale('log')
	ax1.set_yscale('log')

	CS = ax1.contour(Klist,Lambdalist,k_24_lower,colors='k')#,[-1.5,-1,-0.5,-0.2,-0.1,-0.01],colors='k')
	ax1.clabel(CS, inline=1, fontsize=14)

	CS = ax1.contour(Klist,Lambdalist,finaltwist,[0.31],colors='xkcd:orange',linewidths=3)
	ax1.text(40,0.0009,'$\psi_\infty \leq 0.31$',fontsize=20,rotation = -19)
	ax1.contourf(Klist,Lambdalist,finaltwist,[0.31,0.7],alpha=0.5,colors='orange')


	ax1.minorticks_on()
	ax1.tick_params(axis='x', labelsize=16)
	ax1.tick_params(axis='y', labelsize=16)

	#ax2.imshow(k_24_upper,cmap=cmap.autumn,origin='lower',extent=[0,3,-3,2],aspect=3/5)#,interpolation='gaussian')
	ax2.set_xlabel('$ K$',fontsize=20)
	ax2.set_ylabel('$\Lambda$',fontsize=20)
	ax2.set_title("$k_{24}^{max}$",fontsize=20)
	ax2.set_xscale('log')
	ax2.set_yscale('log')

	CS = ax2.contour(Klist,Lambdalist,k_24_upper,colors='k')#,[-1.5,-1,-0.5,-0.2,-0.1,-0.01],colors='k')
	ax2.clabel(CS, inline=1, fontsize=14)

	CS = ax2.contour(Klist,Lambdalist,finaltwist,[0.31],colors='xkcd:orange',linewidths=3)
	ax2.text(40,0.0009,'$\psi_\infty \leq 0.31$',fontsize=20,rotation = -19)
	ax2.contourf(Klist,Lambdalist,finaltwist,[0.31,0.7],alpha=0.5,colors='orange')

	ax2.minorticks_on()
	ax2.tick_params(axis='x', labelsize=16)
	ax2.tick_params(axis='y', labelsize=16)
	plt.tight_layout(pad=0.5)

	plt.show()




# R_0 Figure
def R_0Figure():
	#plt.imshow(r0,cmap=cmap.autumn,origin='lower',extent=[0,3,-3,2],aspect=3/5,interpolation='gaussian')
	plt.xlabel('$\log_{10} K$',fontsize=20)
	plt.ylabel('$\log_{10}\Lambda$',fontsize=20)
	#plt.title('$R_0$',fontsize=20)

	CS = plt.contour(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.09],colors='blue',linewidths=15,alpha=0.5)
	plt.clabel(CS,fmt='Tendon', inline=1, fontsize=16)

	plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.31,0.7],alpha=0.5,colors='orange')
	#plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.08,0.1],alpha=0.5,colors='blue')
	plt.text(0.15,-1.8,'Cornea',color='black',fontsize=16)
	plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.01,0.31],alpha=0.1,colors='grey')

	CS = plt.contour(np.log10(Klist),np.log10(Lambdalist),blurredr0,[0.15,0.5,1,5,10,20],colors='k')
	plt.clabel(CS, inline=1, fontsize=14)

	plt.minorticks_on()
	plt.tick_params(axis='x', labelsize=14)
	plt.tick_params(axis='y', labelsize=14)
	plt.tight_layout(pad=0.5)

	plt.show()



def Plot_RC_over_R0():
	#plt.imshow(maxdeltanovern,cmap=cmap.autumn,origin='lower',extent=[1,10**2.5,0.001,0.1],aspect=0.8,interpolation='gaussian')
	plt.xlabel('$K$',fontsize=20)
	plt.ylabel('$\Lambda$',fontsize=20)
	plt.xscale('log')
	plt.yscale('log')
	#plt.title("max{$\Delta n/n$}",fontsize=20)

	CS = plt.contour(Klist,Lambdalist,finaltwist,[0.31],colors='xkcd:orange',linewidths=3)
	plt.text(30,0.006,'$\psi_\infty \leq 0.31$',fontsize=20,rotation = -38)
	plt.contourf(Klist,Lambdalist,finaltwist,[0.31,0.7],alpha=0.5,colors='orange')

	CS = plt.contour(Klist,Lambdalist,R_C/r0,[0.1,0.2,0.5,1,2,5],colors='k')
	plt.clabel(CS, inline=1, fontsize=20)

	plt.minorticks_on()
	plt.tick_params(axis='x', labelsize=14)
	plt.tick_params(axis='y', labelsize=14)
	plt.tight_layout(pad=0.5)

	plt.show()



def PlotConcentrationControl_justdeltar():
	#plt.imshow(maxdeltanovern,cmap=cmap.autumn,origin='lower',extent=[1,10**2.5,0.001,0.1],aspect=0.8,interpolation='gaussian')
	plt.xlabel('$K$',fontsize=20)
	plt.ylabel('$\Lambda$',fontsize=20)
	plt.xscale('log')
	plt.yscale('log')
	#plt.title("max{$\Delta n/n$}",fontsize=20)

	CS = plt.contour(Klist,Lambdalist,finaltwist,[0.31],colors='xkcd:orange',linewidths=3)
	plt.text(30,0.006,'$\psi_\infty \leq 0.31$',fontsize=20,rotation = -38)
	plt.contourf(Klist,Lambdalist,finaltwist,[0.31,0.7],alpha=0.5,colors='orange')

	CS = plt.contour(Klist,Lambdalist,maxdeltanovern_justdeltar)#,[0.05,0.12,0.16,0.2,0.23],colors='k')
	plt.clabel(CS, inline=1, fontsize=20)

	plt.minorticks_on()
	plt.tick_params(axis='x', labelsize=14)
	plt.tick_params(axis='y', labelsize=14)
	plt.tight_layout(pad=0.5)

	plt.show()



def PlotFPrime():
	plt.xlabel('$K$',fontsize=20)
	plt.ylabel('$\Lambda$',fontsize=20)
	plt.xscale('log')
	plt.yscale('log')
	#plt.title("max{$\Delta n/n$}",fontsize=20)

	CS = plt.contour(Klist,Lambdalist,finaltwist,[0.31],colors='xkcd:orange',linewidths=3)
	plt.text(30,0.006,'$\psi_\infty \leq 0.31$',fontsize=20,rotation = -38)
	plt.contourf(Klist,Lambdalist,finaltwist,[0.31,0.7],alpha=0.5,colors='orange')

	CS = plt.contour(Klist,Lambdalist,fprime_rc,[0.001,0.005,0.01,0.02,0.04,0.06,0.08,0.10,0.12],colors='k')
	plt.clabel(CS, inline=1, fontsize=20)

	plt.minorticks_on()
	plt.tick_params(axis='x', labelsize=16)
	plt.tick_params(axis='y', labelsize=16)
	plt.tight_layout(pad=0.5)

	plt.show()



