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




##### Calculate Twist function and free energy for Lambda = 0 limit
def F0(psi,gr,K):
    return gr - (1/2)*np.sin(2*psi) - K*np.tan(2*psi)*np.sin(psi)**2

def psi0function(gr,K):
    psilist=np.zeros(len(gr))

    for i in range(len(gr)):
        psilist[i] = fsolve(F0,psilist[i-1],args=(gr[i],K))
  
    return psilist

def FreeEnergy_NoD(gr,K,psi,gamma,omega):
    N=len(gr)

    fgamma = np.zeros(N)
    fK = np.zeros(N)
    
    fgamma = gamma/gr
    for i in range(N):
        fK[i] = (1/2)*(np.sin(2*psi[i])/(2*gr[i]))**2 -(np.sin(2*psi[i])/(2*gr[i])) + (1/2)*K*((np.sin(psi[i])**4)/(gr[i]**2))
    f = fgamma + fK + -omega/2
    
    return f

def eta0function(gr,psi):
	etalist = np.zeros(len(gr))
	for i in range(len(gr)):
		if i==0:
			quadint2 = (gr[i])*(gr[i]*np.cos(psi[i])**2)/2
			quadint4 = (gr[i])*(gr[i]*np.cos(psi[i])**4)/2
		else:
			quadint2 += (gr[i]-gr[i-1]) * (gr[i]*np.cos(psi[i])**2 + gr[i-1]*np.cos(psi[i-1])**2)/2
			quadint4 += (gr[i]-gr[i-1]) * (gr[i]*np.cos(psi[i])**4 + gr[i-1]*np.cos(psi[i-1])**4)/2

		etalist[i] = np.sqrt(4*(np.pi**2)*quadint2/quadint4)
	return etalist



##### Params:
K=100
Lambda=0.5
omega=0.1
gamma=0.01
gr = np.logspace(-2,3,num=1000)


# Needs access to TendonData Folder
psi=np.loadtxt('Psi.csv')
etalist=np.loadtxt('Eta.csv')
deltalist=np.loadtxt('Delta.csv')
f=np.loadtxt('F.csv')



##### Plot Basic Solution Figure:
def PlotData(gr,K,Lambda,omega,gamma,psi,deltalist,etalist,f):

	psi_noD = psi0function(gr,K)
	f_noD = FreeEnergy_NoD(gr,K,psi_noD,gamma,omega)
	eta_noD = eta0function(gr,psi_noD)

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
	ax1.set_ylim(0.008,2)
	ax1.set_xlim(0.01,1000)
	#ax1.vlines(gr[np.where(deltalist == min(deltalist))], 0.007,np.pi/4+1,label='$R_0$',color='red',linestyle='--')
	#ax1.arrow(gr[np.where(deltalist == min(deltalist))],0.03,0,-0.017,color='red',width=0.02,head_width=0.3,head_length=0.005)
	ax1.scatter(gr[np.where(deltalist == min(deltalist))],0.01,marker='v',color='red')
	ax1.set_title('A)',loc='left',fontsize=16)
	ax1.hlines(psi[-1],0.05,10000,linestyle=':',color='blue')#,label='$\psi_\infty$'
	ax1.hlines(np.pi/4,0.05,10000,linestyle=':',color='xkcd:orange')
	ax1.text(0.015,psi[-1],'$\psi_\infty$',ha='left', va='center',color='blue',fontsize=14)
	ax1.text(0.015,np.pi/4,'$\pi/4$',ha='left', va='center',color='xkcd:orange',fontsize=14)
	ax1.legend(loc='best',fontsize=14)

	ax2.plot(gr,deltalist, lw=3)
	ax2.plot(gr,np.ones(len(gr)),lw=2,linestyle='--')
	ax2.set_xscale('log')
	ax2.set_ylabel('$\delta$',fontsize=20)
	ax2.set_xlabel('$R$',fontsize=20)
	#ax2.set_xlabel('$r$')
	ax2.set_xlim(0.01,1000)
	ax2.set_title('B)',loc='left',fontsize=16)
	#ax2.vlines(gr[np.where(deltalist == min(deltalist))], 0.95,1.01,label='$R_0$',color='red',linestyle='--')
	ax2.scatter(gr[np.where(deltalist == min(deltalist))],0.9818,marker='v',color='red')
	#ax2.arrow(gr[np.where(deltalist == min(deltalist))],0.9895,0,-0.0045,color='red',width=0.02,head_width=0.3,head_length = 0.002)
	ax2.text(gr[np.where(deltalist == min(deltalist))],0.9825,'$R_0$',color='red',fontsize=14,ha='center')
	ax2.set_ylim(0.981,1.0006)
	ax2.tick_params(axis='x', labelsize=14)
	ax2.tick_params(axis='y', labelsize=14)

	ax3.plot(gr,2*np.pi/etalist, lw=3)
	ax3.plot(gr,2*np.pi/eta_noD,lw=2,linestyle='--')
	ax3.set_xscale('log')
	ax3.set_ylabel('$2\pi/\eta$',fontsize=20)
	ax3.set_xlabel('$R$',fontsize=20)
	ax3.set_ylim(0.985,1.0002)
	ax3.set_xlim(0.01,1000)
	#ax3.vlines(gr[np.where(deltalist == min(deltalist))], 0.99,1.0005,label='$R_0$',color='red',linestyle='--')
	#ax3.arrow(gr[np.where(deltalist == min(deltalist))],0.9935,0,-0.0024,color='red',width=0.02,head_width=0.3,head_length = 0.0011)
	ax3.scatter(gr[np.where(deltalist == min(deltalist))],0.9856,marker='v',color='red')
	#ax3.set_ylim(,1.01)
	ax3.set_title('C)',loc='left',fontsize=16)
	ax3.tick_params(axis='x', labelsize=14)
	ax3.tick_params(axis='y', labelsize=14)

	ax4.plot(gr, f, lw=3,label='Full Solution')
	ax4.plot(gr,f_noD,lw=2,label='$\delta=0$ Solution',linestyle='--')
	ax4.set_xscale('log')
	ax4.set_ylabel('$f$',fontsize=20)
	ax4.set_xlabel('$R$',fontsize=20)
	#ax4.arrow(gr[np.where(deltalist == min(deltalist))],0.9935,0,-0.0024,color='red',width=0.02,head_width=0.3,head_length = 0.0011)
	ax4.scatter(gr[np.where(deltalist == min(deltalist))],-0.32,marker='v',color='red')
	ax4.set_ylim(-0.35,0.55)
	ax4.set_xlim(0.01,1000)
	#ax4.legend(loc='best')
	ax4.set_title('D)',loc='left',fontsize=16)
	ax4.tick_params(axis='x', labelsize=14)
	ax4.tick_params(axis='y', labelsize=14)


	plt.tight_layout(pad=0.5)

	plt.show()



PlotData(gr,K,Lambda,omega,gamma,psi,deltalist,etalist,f)


