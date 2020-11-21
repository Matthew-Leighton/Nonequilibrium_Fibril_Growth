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


# Params used for the data
N=1000
n=20
Klist=np.logspace(0,2.5,num=n)
Lambdalist=np.logspace(-3,-1,num=n)


##### Data Analysis


finaltwist = np.zeros((n,n)) #psi_infty
R_C = np.zeros((n,n)) #R_C
K_matrix = np.zeros((n,n))
twopi_eta_C=np.zeros((n,n))


for j in range(n):
    for i in range(n):
        K_matrix[i,j] = Klist[j]

        deltalist = np.loadtxt('Delta_'+str(j)+'_'+str(i)+'.csv')
        etalist = np.loadtxt('Eta_'+str(j)+'_'+str(i)+'.csv')
        psi = np.loadtxt('Psi_'+str(j)+'_'+str(i)+'.csv')
        gr = np.logspace(-2,3,num=len(psi))

        
        biggerthancornea = np.where(psi>=0.31)[0]
        if len(biggerthancornea)>0:
            R_C[i,j] = gr[min(biggerthancornea)]
            twopi_eta_C[i,j]=2*np.pi/etalist[min(biggerthancornea)]
        else:
            R_C[i,j] = np.NaN
            twopi_eta_C[i,j]=np.NaN
        
        if np.where(psi == max(psi))[0][0]<len(psi)-1:
            finaltwist[i,j]=np.NaN
        else:
            finaltwist[i,j] = psi[-1]


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

nf = 1.32*10**24

k_24_lower = 1 - (qupper*tilde_R_C/R_C)
k_24_upper = 1 - (qlower*tilde_R_C/R_C)


tilde_R_C = 15*10**(-9) # m
X = tilde_R_C/R_C *10**9
K33bound = K_matrix * (1-k_24_upper**2)
K33 = K_matrix * ( (1- (1 -qupper*tilde_R_C/R_C)**2))


def PlotData():

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

    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.set_ylabel('$\Lambda$',fontsize=20)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('$K$',fontsize=20)
    ax1.set_title('A)',loc='left',fontsize=16)
    ax1.set_title('$K_{33}^C$',loc='right',fontsize=20)
    CS = ax1.contour(Klist,Lambdalist,finaltwist,[0.31],colors='xkcd:orange',linewidths=3)
    ax1.text(23,0.005,'$\psi_\infty \leq 0.31$',fontsize=20,rotation = -35)
    ax1.contourf(Klist,Lambdalist,finaltwist,[0.31,0.7],alpha=0.5,colors='orange')
    #ax1.legend(loc='best')

    CS = ax1.contour(Klist,Lambdalist,K33,[0.4,0.8,1,1.2,1.35],colors='k')
    ax1.clabel(CS, inline=1, fontsize=14)


    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylabel('$\Lambda$',fontsize=20)
    ax2.set_xlabel('$K$',fontsize=20)
    ax2.set_title('B)',loc='left',fontsize=16)
    ax2.set_title('$2\pi/\eta_C$',loc='right',fontsize=20)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    CS = ax2.contour(Klist,Lambdalist,finaltwist,[0.31],colors='xkcd:orange',linewidths=3)
    #ax2.text(30,0.006,'$\psi_\infty \leq 0.31$',fontsize=20,rotation = -38)
    ax2.contourf(Klist,Lambdalist,finaltwist,[0.31,0.7],alpha=0.5,colors='orange')

    CS = ax2.contour(Klist,Lambdalist,twopi_eta_C,[0.96,0.963,0.966,0.969,0.972],colors='k')
    ax2.clabel(CS, inline=1, fontsize=14)



    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_ylabel('$\Lambda$',fontsize=20)
    ax3.set_xlabel('$K$',fontsize=20)
    ax3.set_title('C)',loc='left',fontsize=16)
    ax3.set_title('$k_{24}^C$',loc='right',fontsize=20)
    ax3.tick_params(axis='x', labelsize=14)
    ax3.tick_params(axis='y', labelsize=14)
    CS = ax3.contour(Klist,Lambdalist,k_24_lower,[0.8,0.84,0.88,0.92,0.96,0.98,0.99],colors='k')#,[-1.5,-1,-0.5,-0.2,-0.1,-0.01],colors='k')
    ax3.clabel(CS, inline=1, fontsize=14)

    CS = ax3.contour(Klist,Lambdalist,finaltwist,[0.31],colors='xkcd:orange',linewidths=3)
    ax3.contourf(Klist,Lambdalist,finaltwist,[0.31,0.7],alpha=0.5,colors='orange')



    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_ylabel('$\Lambda$',fontsize=20)
    ax4.set_xlabel('$K$',fontsize=20)
    ax4.set_title('D)',loc='left',fontsize=16)
    ax4.set_title('$\\tilde{R}_C/R_C$ (nm)',loc='right',fontsize=20)
    ax4.tick_params(axis='x', labelsize=14)
    ax4.tick_params(axis='y', labelsize=14)
    CS = ax4.contour(Klist,Lambdalist,finaltwist,[0.31],colors='xkcd:orange',linewidths=3)
    #ax4.text(30,0.006,'$\psi_\infty \leq 0.31$',fontsize=20,rotation = -38)
    ax4.contourf(Klist,Lambdalist,finaltwist,[0.31,0.7],alpha=0.5,colors='orange')

    CS = ax4.contour(Klist,Lambdalist,X,[1,2.5,5,7.5,10,12.5],colors='k')
    ax4.clabel(CS, inline=1, fontsize=14)



    plt.tight_layout(pad=0.5)

    plt.show()

PlotData()
