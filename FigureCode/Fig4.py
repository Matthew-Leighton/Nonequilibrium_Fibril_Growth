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



# Needs access to TendonData Folder
f=np.loadtxt('F.csv')

##### Params used in data:
K=500
Lambda=0.5
omega=0.1
gamma=0.01
gr = np.logspace(-2,3,num=1000)



##### Plot Radius Control Figure 4:

def PlotFEDensity(gr,f):
	plt.plot(gr, f, lw=3,zorder=0,label = '$f(R)$')
	plt.hlines(-0.15,0.01,1000,linestyle='--',color='red',lw = 2,zorder=1,label = '$n_f\cdot \mu$')
	plt.scatter(0.85,-0.148,marker='*',color='red',s=500,zorder=2)
	#plt.scatter(10,-0.01,marker=11,color='red',s=300,zorder=3)
	#plt.vlines(0.75,-0.4,-0.29,linestyle=':',color='red',zorder=4)
	#plt.text(0.6,-0.285,'$R_0$',color='red',fontsize=16,zorder=4)
	plt.scatter(0.75,-0.33,marker='v',s=150,color='red')
	#plt.scatter(45,-0.11,marker='$\mu$',color='red',s=400,zorder=5)
	plt.xscale('log')
	plt.xlabel('$R$',fontsize=20)
	plt.ylabel('$f$',fontsize=20)
	plt.xlim(0.01,1000)
	plt.ylim(-0.35,0.2)
	plt.minorticks_on()
	plt.tick_params(axis='x', labelsize=16)
	plt.tick_params(axis='y', labelsize=16)
	plt.tight_layout(pad=0.5)
	plt.legend(loc='best',fontsize=20)

	plt.show()



PlotFEDensity(gr,f)


