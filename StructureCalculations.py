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

def F(psi,gr,K,Lambda,delta,eta):
    return np.abs(gr - (1/2)*np.sin(2*psi) - K*np.tan(2*psi)*np.sin(psi)**2 - Lambda*((4*np.pi**2) - (eta**2) * np.cos(psi)**2)*np.tan(2*psi)*delta*(eta*gr)**2)

def psifunction(gr,K,deltalist,etalist,Lambda):
    psilist=np.zeros(len(gr))
    for i in range(len(gr)):
        psilist[i] = minimize(F,psilist[i-1],args=(gr[i],K,Lambda,deltalist[i],etalist[i])).x
    return psilist

def F0(psi,gr,K):
    return gr - (1/2)*np.sin(2*psi) - K*np.tan(2*psi)*np.sin(psi)**2

def psi0function(gr,K):
    psilist=np.zeros(len(gr))

    for i in range(len(gr)):
        psilist[i] = fsolve(F0,psilist[i-1],args=(gr[i],K))
  
    return psilist


def CalculateStructure(gr,K,Lambda,omega):
	N = len(gr)

	psi=np.zeros(N)
	etalist=np.zeros(N)
	deltalist=np.ones(N)

	iterations = 0
	while iterations<100: #np.sum(abs(olddelta - deltalist))>0.001:
		iterations+=1

		psi = psifunction(gr,K,deltalist,etalist,Lambda)

		for i in range(N):
			if i==0:
				quadint2 = (gr[i])*(gr[i]*np.cos(psi[i])**2)/2
				quadint4 = (gr[i])*(gr[i]*np.cos(psi[i])**4)/2
			else:
				quadint2 += (gr[i]-gr[i-1]) * (gr[i]*np.cos(psi[i])**2 + gr[i-1]*np.cos(psi[i-1])**2)/2
				quadint4 += (gr[i]-gr[i-1]) * (gr[i]*np.cos(psi[i])**4 + gr[i-1]*np.cos(psi[i-1])**4)/2

			etalist[i] = np.sqrt(4*(np.pi**2)*quadint2/quadint4)
			deltalist[i] =  ( 1 - (8*np.pi**4 * Lambda/omega) + (4*np.pi**2 * (Lambda/omega)*(1/(gr[i]**2))*etalist[i]**2 * quadint2) )#**(1/2)
			#deltalist[i]= m.sqrt(deltalist[i])

	return psi,etalist,deltalist


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

	return f,fK,fLambda,fomega,fgamma


def FreeEnergy_NoD(gr,K,psi,gamma):
    N=len(gr)

    fgamma = np.zeros(N)
    fK = np.zeros(N)
    
    fgamma = gamma/gr
    for i in range(N):
        fK[i] = (1/2)*(np.sin(2*psi[i])/(2*gr[i]))**2 -(np.sin(2*psi[i])/(2*gr[i])) + (1/2)*K*((np.sin(psi[i])**4)/(gr[i]**2))
    f = fgamma + fK
    
    return f





def PlotData(gr,K,Lambda,omega,gamma,psi,deltalist,etalist,f):

	psi_noD = psi0function(gr,K)
	f_noD = FreeEnergy_NoD(gr,K,psi_noD,gamma)

	fig=plt.figure()
	gs=gridspec.GridSpec(2,2,width_ratios=[1,1],height_ratios=[1,1])
	#,wspace =0.8,top=0.7,bottom=0.3,left=0.1,right=0.9)

	ax1=plt.subplot(gs[0])
	ax2=plt.subplot(gs[1])
	ax3=plt.subplot(gs[2])
	ax4=plt.subplot(gs[3])

	ax1.plot(gr, psi, lw=2,label='Full Solution')
	ax1.plot(gr,psi_noD,lw=2,label='$\delta=0$ Solution',linestyle='--')
	#ax1.set_xlabel('$r$')
	ax1.set_ylabel('$\psi$',fontsize=14)
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.set_ylim(0,np.pi/4)
	ax1.legend()
	ax1.set_title('A)',loc='left',fontsize=16)
	#ax1.set_title('$K=$'+str(K)+', $\gamma = $'+str(gamma)+', $\Lambda=$'+str(Lambda)+', and $\omega=$'+str(omega))

	ax2.plot(gr,deltalist)
	ax2.set_xscale('log')
	ax2.set_ylabel('$\delta$',fontsize=14)
	#ax2.set_xlabel('$r$')
	#ax2.set_ylim(0,1.01)
	ax2.set_title('B)',loc='left',fontsize=16)

	ax3.plot(gr,2*np.pi/etalist)
	ax3.set_xscale('log')
	ax3.set_ylabel('$2\pi/\eta$',fontsize=14)
	ax3.set_xlabel('$r$',fontsize=14)
	#ax3.set_ylim(,1.01)
	ax3.set_title('C)',loc='left',fontsize=16)

	ax4.plot(gr, f, lw=2,label='Full Solution')
	ax4.plot(gr,f_noD,lw=2,label='$\delta=0$ Solution',linestyle='--')
	ax4.set_xscale('log')
	ax4.set_ylabel('$f$',fontsize=14)
	ax4.set_xlabel('$r$',fontsize=14)
	#ax4.set_ylim(-0.6,0.5)
	ax4.legend(loc='best')
	ax4.set_title('D)',loc='left',fontsize=16)

	plt.show()



def PlotMolecularStrain(gr,psi,etalist):
	molecularstrain = (2*np.pi/etalist - np.cos(psi))/np.cos(psi)
	plt.plot(gr,molecularstrain*100)
	plt.xlabel('r',fontsize=14)
	plt.ylabel('Molecular Strain (%)',fontsize=14)
	plt.xscale('log')
	#plt.legend(loc='best')

	plt.show()



