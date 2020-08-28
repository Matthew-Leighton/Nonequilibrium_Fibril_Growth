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
	while iterations<200: #np.sum(abs(olddelta - deltalist))>0.001:
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
			#deltalist[i]= np.sqrt(max(deltalist[i],0))
			deltalist[i] = np.sqrt(deltalist[i])

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

	ax1.minorticks_on()
	ax2.minorticks_on()
	ax3.minorticks_on()
	ax4.minorticks_on()

	ax1.plot(gr, psi,label='Full Solution', lw=3)
	ax1.plot(gr,psi_noD,lw=2,label='$\Lambda=0$',linestyle='--')
	ax1.tick_params(axis='x', labelsize=14)
	ax1.tick_params(axis='y', labelsize=14)
	#ax1.set_xlabel('$r$')
	ax1.set_ylabel('$\psi$',fontsize=20)
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.set_xlabel('$r$',fontsize=20)
	ax1.set_ylim(0.008,np.pi/4+0.1)
	ax1.set_xlim(0.01,1000)
	ax1.vlines(gr[np.where(deltalist == min(deltalist))], 0.007,np.pi/4+1,label='$R_0$',color='red',linestyle='--')
	ax1.set_title('A)',loc='left',fontsize=16)
	ax1.hlines(psi[-1],0.001,10000,linestyle=':',color='blue')#,label='$\psi_\infty$'
	ax1.hlines(np.pi/4,0.001,10000,linestyle=':',color='orange')#,label='$\pi/4$'
	ax1.text(0.02,0.07,'$\psi_\infty$',color='blue',fontsize=14)
	ax1.text(0.02,0.45,'$\pi/2$',color='orange',fontsize=14)
	ax1.legend(loc='best',fontsize=12)

	ax2.plot(gr,deltalist, lw=3)
	ax2.set_xscale('log')
	ax2.set_ylabel('$\delta$',fontsize=20)
	ax2.set_xlabel('$r$',fontsize=20)
	#ax2.set_xlabel('$r$')
	ax2.set_xlim(0.01,1000)
	ax2.set_title('B)',loc='left',fontsize=16)
	ax2.vlines(gr[np.where(deltalist == min(deltalist))], 0.95,1.01,label='$R_0$',color='red',linestyle='--')
	ax2.set_ylim(0.984,1.0006)
	ax2.tick_params(axis='x', labelsize=14)
	ax2.tick_params(axis='y', labelsize=14)

	ax3.plot(gr,2*np.pi/etalist, lw=3)
	ax3.set_xscale('log')
	ax3.set_ylabel('$2\pi/\eta$',fontsize=20)
	ax3.set_xlabel('$r$',fontsize=20)
	ax3.set_ylim(0.99,1.0002)
	ax3.set_xlim(0.01,1000)
	ax3.vlines(gr[np.where(deltalist == min(deltalist))], 0.99,1.0005,label='$R_0$',color='red',linestyle='--')
	#ax3.set_ylim(,1.01)
	ax3.set_title('C)',loc='left',fontsize=16)
	ax3.tick_params(axis='x', labelsize=14)
	ax3.tick_params(axis='y', labelsize=14)

	ax4.plot(gr, f, lw=3,label='Full Solution')
	ax4.plot(gr,f_noD,lw=2,label='$\delta=0$ Solution',linestyle='--')
	ax4.set_xscale('log')
	ax4.set_ylabel('$f$',fontsize=20)
	ax4.set_xlabel('$r$',fontsize=20)
	ax4.vlines(gr[np.where(deltalist == min(deltalist))], -0.4,0.6,label='$R_0$',color='red',linestyle='--')
	ax4.set_ylim(-0.35,0.55)
	ax4.set_xlim(0.01,1000)
	#ax4.legend(loc='best')
	ax4.set_title('D)',loc='left',fontsize=16)
	ax4.tick_params(axis='x', labelsize=14)
	ax4.tick_params(axis='y', labelsize=14)


	plt.tight_layout(pad=0.5)

	plt.show()



def PlotMolecularStrain(gr,psi,etalist):

	molecularstrain = ((2*np.pi/etalist[-1] - np.cos(psi))/np.cos(psi))
	molecularstrainsmall = ((2*np.pi/etalist[330] - np.cos(psi[:330]))/np.cos(psi[:330]))

	plt.plot(gr,molecularstrain*100,label='$R \gg R_0$')
	plt.plot(gr[:330],molecularstrainsmall*100,label='$R \ll R_0$')

	plt.xlabel('r',fontsize=14)
	plt.ylabel('Molecular Strain (%)',fontsize=14)
	plt.xscale('log')
	plt.legend(loc='best')

	plt.show()

def MolecularStrainCrossSection(gr,psi,etalist):
	#gr = gr[:200]
	#psi=psi[:200]

	molecularstrain = ((2*np.pi/etalist[-1] - np.cos(psi))/np.cos(psi))
	azm = np.linspace(0, 2 * np.pi, len(gr))
	fig = plt.figure()
	ax = Axes3D(fig)
	r, th = np.meshgrid(gr, azm)

	data= np.zeros((len(gr),len(gr)))
	for i in range(len(gr)):
		data[i,:] = molecularstrain

	plt.subplot(projection="polar")
	plt.pcolormesh(th, r, data)
	#plt.plot(th, r, color='k', ls='none') 
	#plt.grid()
	plt.colorbar()
	ax.set_yticks([])
	ax.set_xticks([])

	plt.show()

def MolecularStrainCrossSectionSmall(gr,psi,etalist):
	gr = gr[:330]
	psi=psi[:330]
	#etalist = etalist[:330]

	molecularstrain = ((2*np.pi/etalist[330] - np.cos(psi))/np.cos(psi))
	azm = np.linspace(0, 2 * np.pi, len(gr))
	fig = plt.figure()
	ax = Axes3D(fig)
	r, th = np.meshgrid(gr, azm)

	data= np.zeros((len(gr),len(gr)))
	for i in range(len(gr)):
		data[i,:] = molecularstrain

	plt.subplot(projection="polar")
	plt.pcolormesh(th, r, data)
	#plt.plot(th, r, color='k', ls='none') 
	#plt.grid()
	plt.colorbar()
	#ax.set_yticks([])
	#ax.set_xticks([])

	plt.show()


def PlotFEDensity(gr,f):
	plt.plot(gr, f, lw=3,zorder=0,label = '$f(r)$')
	plt.hlines(-0.15,0.01,1000,linestyle='--',color='red',lw = 2,zorder=1,label = '$n_f\cdot \mu$')
	plt.scatter(0.85,-0.148,marker='*',color='red',s=300,zorder=2)
	plt.scatter(10,-0.01,marker=11,color='red',s=300,zorder=3)
	plt.vlines(0.75,-0.4,-0.29,linestyle=':',color='red',zorder=4)
	plt.scatter(0.75,-0.265,marker='$R_0$',color='red',s=300,zorder=4)
	#plt.scatter(45,-0.11,marker='$\mu$',color='red',s=400,zorder=5)
	plt.xscale('log')
	plt.xlabel('$r$',fontsize=20)
	plt.ylabel('$f$',fontsize=20)
	plt.xlim(0.01,62)
	plt.ylim(-0.35,0.5)
	plt.minorticks_on()
	plt.tick_params(axis='x', labelsize=14)
	plt.tick_params(axis='y', labelsize=14)
	plt.tight_layout(pad=0.5)
	plt.legend(loc='best',fontsize=20)

	plt.show()



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

	#plt.imshow(finaltwist,cmap=cmap.autumn,origin='lower',extent=[0,3,-3,2],aspect=3/5,interpolation='gaussian')
	plt.xlabel('$\log_{10} K$',fontsize=20)
	plt.ylabel('$\log_{10}\Lambda$',fontsize=20)

	CS = plt.contour(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.06,0.16,0.24,0.31,0.4,0.48,0.56],colors='k')
	plt.clabel(CS, inline=1, fontsize=12)

	CS = plt.contour(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.09],colors='k')
	plt.clabel(CS,fmt='Tendon', inline=1, fontsize=16)

	plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.31,0.7],alpha=0.5,colors='orange')
	plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.08,0.1],alpha=0.5,colors='blue')
	plt.text(0.35,-1.8,'Cornea',color='black',fontsize=16)
	plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.1,0.31],alpha=0.1,colors='grey')
	plt.contourf(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.01,0.08],alpha=0.1,colors='grey')

	#CS = plt.contour(np.log10(Klist),np.log10(Lambdalist),finaltwist,[0.31],colors='k')
	#plt.clabel(CS,fmt='Cornea', inline=1, fontsize=12)

	plt.minorticks_on()
	plt.tick_params(axis='x', labelsize=14)
	plt.tick_params(axis='y', labelsize=14)
	plt.tight_layout(pad=0.5)

	plt.show()


