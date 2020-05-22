import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from my_plot import set_size
from scipy.stats import linregress

#Settings#####################
nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": False,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
}
mpl.rcParams.update(nice_fonts)
###############################
print('Settings updated')





#Figure Plotting######################

#Wake_bck_multiple#
q = np.array(map(int,np.linspace(0,len(Tgrid)-1,3)))
qreal=Tgrid[q]/(2*np.pi)
eks=Xgrid/(2*np.pi)
fig, axs = plt.subplots(3,1, figsize=set_size(subplots=(1.,1.)))
# Plot
for i in range(len(q)):
 axs[i].axes.xaxis.set_visible(True)
 if i == range(len(q))[-1]:
  axs[i].set_xlabel('$x/\lambda_d$')
 axs[i].set_ylabel('$E/E^0_s$')
 axs[i].plot(eks,Eout[q[i]])
 axs[i].plot(eks,8*np.interp(Xgrid-vphi*Tgrid[q[i]]-push,ksigrid,Profile),'r',label='$n_e/\gamma_e$')
 axs[i].set_title('$t = %.2f/\omega_d$' %qreal[i])
 axs[i].legend(loc='lower right')

fig.tight_layout()
plt.show()
#fig.savefig('/home/christoffer/Thesis_MCs_figures/plasmaprop.pdf', format='pdf', bbox_inches='tight')


#Energy gain in space####
q = np.array(map(int,np.linspace(0,len(Tgrid)-1,3)))
qreal=Tgrid[q]/(2*np.pi)
eks=Xgrid/(2*np.pi)
Ez=Eout
By=Bout
Eseed = lambda EZ,BY: (1./2)*integrate.cumtrapz(EZ**2+BY**2,Xgrid,initial=0.)
fig, axs = plt.subplots(1, 3, figsize=set_size(subplots=(1.,1.)))
# Plot
for i in range(len(q)):
 if i == 0:
  axs[i].set_ylabel('Seed Field Energy')
 axs[i].plot(eks,Eseed(Eout[q[i]],Bout[q[i]]))
 axs[i].plot(eks,10*Eout[q[i]],'c',label='Seed field')
 axs[i].plot([],[],'w',label=
 axs[i].set_xlabel('$x/ \lambda_d$')
 axs[i].set_title('$t = %.2f/\omega_d$' %qreal[i])
 axs[i].legend(loc='lower right')

fig.tight_layout()
plt.show()
#fig.savefig('/home/christoffer/Thesis_MCs_figures/plasmaprop.pdf', format='pdf', bbox_inches='tight')

#Energy gain in time####
eks=Xgrid/(2*np.pi)
teks=Tgrid/(2*np.pi)
Ez=Eout
By=Bout
Eseed = lambda EZ,BY: (1./2)*np.trapz(EZ**2+BY**2,Xgrid)
Egaint = np.zeros(len(Tgrid))
for i in range(len(Tgrid)):
 Egaint[i] = Eseed(Ez[i],By[i])
fig, ax = plt.subplots(1, 1, figsize=set_size(subplots=(1.,1.)))
ax.plot(teks,Egaint)
ax.set_xlabel('$\omega_d t$')
ax.set_ylabel('Seed Field Energy')

plt.show()
#fig.savefig('/home/christoffer/Thesis_MCs_figures/plasmaprop.pdf', format='pdf', bbox_inches='tight')

#Linear vs Non-linear wakes#
l0 = 2*np.pi #Reference length in this simulation inspired from 1D simulations
t0=l0  #Reference time inspired from 1D simulations
resx=105. #Spatial resolution def 105
rest=120. #Temporal resolution def 120
dt=t0/rest #Temporal grid length
dx=l0/resx #Spatial grid length
start=0. #Spatial grid start
stop=200.*l0 #Spatial grid end
tstart = 0. #Temporal grid start
tstop = 38.5*t0 #Temporal grid stop, default is 45t0
Xgrid = np.arange(start,stop,dx) #Spatial grid with resolution dx
Tgrid = np.arange(tstart,tstop,dt) #Temporal grid with resolution dt
ne=0.026
vphi = 1-(3*ne)/2 #Group velocity of plasma wave
gamme=1./np.sqrt(1-vphi**2)
omegad=1.
vp=1./vphi
kd=omegad/vp
#Wakefield parameters#
ksigrid=np.arange(-100.,300.,dx) #Grid to evaluate model on
kp=np.sqrt(ne) #Plasma wavenumber
a00 = [0.6,2.5]        #Normalized driver amplitude [0.6,2.5] for thesis
ksi0 = 1./(kp/np.sqrt(2)) #Driver RMS width
phiD=0.
push=Xgrid[int(0.4*len(Xgrid))]+40.
#Initial conditions#
y1_0 = 0.0
y2_0 = 0.0
y0 = [y1_0,y2_0]
fig, axs = plt.subplots(1, 2, figsize=set_size(subplots=(1.,2.)))
subtxt = ['(a)','(b)']
for i in [0,1]:
 parameters = [ne,vphi,gamme,kd,ksigrid,kp,a00[i],ksi0,phiD,push]
 axs[i].plot(ksigrid/(2*np.pi),WKF.wake_original(parameters,y0)[0],'k',label='$n/n_e - 1$')
 axs[i].plot(ksigrid/(2*np.pi),WKF.wake_original(parameters,y0)[1]/(np.amax(WKF.wake_original(parameters,y0)[1])),'--k',label='$E_{\phi} / E^0_{\phi}$')
 axs[i].legend(loc='lower right')
 axs[i].set_xlabel('$\\xi / \lambda_d$')
 axs[i].set_ylabel('')
 axs[i].set_xlim(-5,30)
 axs[i].text(0.03,0.9,subtxt[i],transform=axs[i].transAxes)

fig.tight_layout()
plt.show()
#fig.savefig('/home/christoffer/Thesis_MCs_figures/plasmaprop.pdf', format='pdf', bbox_inches='tight')







#Animation#############
for m in range(0,len(Tgrid),50):
  plt.clf()
  plt.plot(Xgrid,Eout[m],'c')
  plt.plot(Xgrid,8*np.interp(Xgrid-vphi*Tgrid[m]-push,ksigrid,Profile),'r')
  plt.pause(0.05)


for m in range(0,len(Tgrid),50):
  plt.clf()
  plt.plot(Xgrid,np.interp(Xgrid-vb*Tgrid[m],ksigrid,WKF.wake_bck(parameters,y0),'r'))
  plt.pause(0.05)



