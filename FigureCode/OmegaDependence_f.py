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

def F(psi,gr,K,Lambda,delta,eta):
    return np.abs(gr - (1/2)*np.sin(2*psi) - K*np.tan(2*psi)*np.sin(psi)**2 - Lambda*((4*np.pi**2) - (eta**2) * np.cos(psi)**2)*np.tan(2*psi)*(delta*eta*gr)**2)

def psifunction(gr,K,deltalist,etalist,Lambda):
    psilist=np.zeros(len(gr))
    for i in range(len(gr)):
        psilist[i] = minimize(F,psilist[i-1],args=(gr[i],K,Lambda,deltalist[i],etalist[i])).x
    return psilist


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
K=100
Lambda=0.5
omega=0.1
gamma=0.01
gr = np.logspace(-2,3,num=1000)

# Needs access to Tendon Data File
psi=np.loadtxt('Psi.csv')
etalist=np.loadtxt('Eta.csv')
deltalist=np.loadtxt('Delta.csv')
f=np.loadtxt('F.csv')

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




##### Plot Radius Control Figure:

def PlotFig(gr,f):
	omegalist=[0.1,1,10,100,1000]
	for i in range(len(omegalist)):
		f = CalculateFreeEnergy(gr,K,Lambda,omegalist[i],gamma,psi,deltalist,etalist)
		plt.plot(gr,f+omegalist[i]/2,label='$\omega=$'+str(omegalist[i]),lw=2.5)

	plt.hlines(0,0.001,10000,ls=':',color='grey')
	plt.xscale('log')
	plt.xlabel('$R$',fontsize=20)
	plt.ylabel('$f+\omega/2$',fontsize=20)
	plt.xlim(0.01,1000)
	#plt.ylim(-0.35,0.2)
	plt.minorticks_on()
	plt.tick_params(axis='x', labelsize=16)
	plt.tick_params(axis='y', labelsize=16)
	plt.tight_layout(pad=0.5)
	plt.legend(loc='best',fontsize=20)

	plt.show()

def PlotFig2(gr,f):
	omegalist=[1/(1000*1000),0.001,1,1000]
	linestylelist = ['-','--','-.',':']
	for i in range(len(omegalist)):
		psi,etalist,deltalist = CalculateStructure_2(gr,K,Lambda,omegalist[i])
		f = CalculateFreeEnergy(gr,K,Lambda,omegalist[i],gamma,psi,deltalist,etalist)
		plt.plot(gr,f+omegalist[i]/2,label='$\omega=$'+str(omegalist[i]),lw=2.5,ls=linestylelist[i])

	plt.hlines(0,0.001,10000,ls=':',color='grey')
	plt.xscale('log')
	plt.xlabel('$R$',fontsize=20)
	plt.ylabel('$f+\omega/2$',fontsize=20)
	plt.xlim(0.01,1000)
	#plt.ylim(-0.35,0.2)
	plt.minorticks_on()
	plt.tick_params(axis='x', labelsize=16)
	plt.tick_params(axis='y', labelsize=16)
	plt.tight_layout(pad=0.5)
	plt.legend(loc='best',fontsize=20)

	plt.show()





def Plotsinglefunction(gr,f):
	omegalist=[1000]
	for i in range(len(omegalist)):
		f = CalculateFreeEnergy(gr,K,Lambda,omegalist[i],gamma,psi,deltalist,etalist)
		plt.plot(gr,f,label='$\omega=$'+str(omegalist[i]),lw=2.5)

	#plt.hlines(0,0.001,10000,ls=':',color='grey')
	plt.xscale('log')
	#plt.yscale('symlog')
	plt.xlabel('$R$',fontsize=20)
	plt.ylabel('$f$',fontsize=20)
	plt.xlim(0.01,1000)
	#plt.ylim(-0.35,0.2)
	plt.minorticks_on()
	plt.tick_params(axis='x', labelsize=16)
	plt.tick_params(axis='y', labelsize=16)
	plt.tight_layout(pad=0.5)
	plt.legend(loc='best',fontsize=20)

	plt.show()



PlotFig2(gr,f)



