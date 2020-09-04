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


##### Params:
K=500
Lambda=0.5
omega=0.1
gamma=0.01
gr = np.logspace(-2,3,num=1000)

# Needs access to Tendon Data File
psi=np.loadtxt('Psi.csv')
etalist=np.loadtxt('Eta.csv')
deltalist=np.loadtxt('Delta.csv')
f=np.loadtxt('F.csv')



##### Plot Molecular Strain Figure:
def PlotMolecularStrain(gr,psi,etalist,deltalist):

	molecularstrain = ((2*np.pi/etalist[-1] - np.cos(psi))/np.cos(psi))
	molecularstrainsmall = ((2*np.pi/etalist[330] - np.cos(psi[:330]))/np.cos(psi[:330]))

	plt.plot(gr,molecularstrain*100,label='$R \gg R_0$',lw=3,color='blue')
	plt.plot(gr[:330],molecularstrainsmall*100,label='$R \ll R_0$',lw=3,color='orange',ls='-.')

	plt.xlabel('$r$',fontsize=20)
	plt.ylabel('Molecular Strain (\%)',fontsize=20)
	plt.xscale('log')
	plt.xlim(0.01,1100)
	plt.ylim(-1.0,0.2)
	plt.scatter(1000,molecularstrain[-1]*100,marker='o',s=300,color='blue')
	plt.scatter(gr[330],molecularstrainsmall[-1]*100,marker='o',s=300,color='orange')
	plt.vlines(gr[np.where(deltalist == min(deltalist))], -1.5,1,label='$R_0$',color='red',linestyle='--',lw=2)

	plt.minorticks_on()
	plt.tick_params(axis='x', labelsize=14)
	plt.tick_params(axis='y', labelsize=14)
	plt.tight_layout(pad=0.5)
	plt.legend(loc='best',fontsize=20)

	plt.show()









