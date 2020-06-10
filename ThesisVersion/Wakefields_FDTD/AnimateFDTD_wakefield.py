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
subtxt=['(a)','(b)','(c)']
# Plot
for i in range(len(q)):
 if i == range(len(q))[-1]:
  axs[i].set_xlabel('$x/\lambda_d$')
 axs[i].set_ylabel('$E_s/E^0_s$')
 axs[i].plot(eks,Eout[q[i]])
 axs[i].set_xlim(50,160)
 axs[i].plot(eks,10*np.interp(Xgrid-vphi*Tgrid[q[i]]-push,ksigrid,Profile),'r',label='$n_e/\gamma_e$')
 axs[i].set_title('$\omega_d t = %.2f$' %qreal[i])
 axs[i].legend(loc='lower right')
 axs[i].text(0.03,0.8,subtxt[i],transform=axs[i].transAxes)

fig.tight_layout()
plt.show()
#fig.savefig('/home/christoffer/Thesis_MCs_figures/plasmaprop.pdf', format='pdf', bbox_inches='tight')

##Wake_bck_single + power spec##
C = lambda t,tc,w: np.exp(-((t-tc)**10)/(w**10))
tc=131.
w=5.
K = np.fft.rfftfreq(n=len(Eout[-1]),d=1./resx)
Signal = lambda(arr): np.abs(np.fft.rfft(arr))**2
kd = 1.
eks=Xgrid/(2*np.pi)
fig, axs = plt.subplots(1,2, figsize=set_size(subplots=(1.,2.)))
subtxt=['(a)','(b)']

axs[0].set_xlabel('$x/\lambda_d$')
axs[0].set_ylabel('$E_s/E^0_s$')
axs[0].plot(eks,Eout[-1])
axs[0].plot(eks,C(eks,tc,w)*np.amax(Eout[-1]),'--k',label='Super-Gaussian filter')
axs[0].set_xlim(50,158)
axs[0].legend(loc='lower right')
axs[0].text(0.03,0.85,subtxt[0],transform=axs[0].transAxes)

axs[1].set_xlabel('$k / k_d$')
axs[1].set_ylabel('$I(\omega) \ (a.u)$')
axs[1].plot(K/kd,Signal(C(eks,tc,w)*Eout[-1]),'c')
axs[1].set_xlim(0,4)
axs[1].text(0.03,0.85,subtxt[1],transform=axs[1].transAxes)

fig.tight_layout()
plt.show()
#fig.savefig('/home/christoffer/Thesis_MCs_figures/plasmaprop.pdf', format='pdf', bbox_inches='tight')

##Wake_bck_multiple + power spec##
C = lambda t,tc,w: np.exp(-((t-tc)**10)/(w**10))
tc=137.
w=18.
K = np.fft.rfftfreq(n=len(Eout[-1]),d=1./resx)
Signal = lambda(arr): np.abs(np.fft.rfft(arr))**2
kd = 1.
eks=Xgrid/(2*np.pi)
fig, axs = plt.subplots(1,2, figsize=set_size(subplots=(1.,2.)))
subtxt=['(a)','(b)']

axs[0].set_xlabel('$x/\lambda_d$')
axs[0].set_ylabel('$E_s/E^0_s$')
axs[0].plot(eks,Eout[-1])
axs[0].plot(eks,C(eks,tc,w)*np.amax(Eout[-1]),'--k',label='Super-Gaussian filter')
axs[0].set_xlim(50,168)
axs[0].legend(loc='lower right')
axs[0].text(0.03,0.85,subtxt[0],transform=axs[0].transAxes)

axs[1].set_xlabel('$k / k_d$')
axs[1].set_ylabel('$I(\omega) \ (a.u)$')
axs[1].plot(K/kd,Signal(C(eks,tc,w)*Eout[-1]),'c')
axs[1].set_xlim(0,4)
axs[1].text(0.03,0.85,subtxt[1],transform=axs[1].transAxes)

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
 axs[i].set_title('$\omega_d t = %.2f$' %qreal[i])
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

#mult wake CEP#
eks=Xgrid/(2*np.pi)
fig, ax = plt.subplots(1, 1, figsize=set_size(subplots=(1.,1.)))

ax.set_ylabel('$E_s/E^0_s$')
ax.plot(eks,Ephase[0],label='$\phi_s = 0$')
ax.plot(eks,np.abs(hilbert(Ephase[0])),'k')
ax.plot(eks,-np.abs(hilbert(Ephase[0])),'k')
ax.plot(eks,Ephase[1],label='$\phi_s = \pi/4$')
ax.plot(eks,Ephase[2],label='$\phi_s = \pi/2$')
ax.plot(eks,Ephase[3],label='$\phi_s = \\frac{3 \pi}{4}$')
ax.set_xlim(87.8,88.8)
ax.legend(loc='lower right')
ax.set_xlabel('$x/ \lambda_d$')

fig.tight_layout()
plt.show()
#fig.savefig('/home/christoffer/Thesis_MCs_figures/vacuumprop.pdf', format='pdf', bbox_inches='tight')

#Mult wake CEP HILBERT ENVELOPE#
eks=Xgrid/(2*np.pi)
fig, ax = plt.subplots(1, 1, figsize=set_size(subplots=(1.,1.)))

ax.set_ylabel('$E_s/E^0_s$')
ax.plot(eks,np.abs(hilbert(Ephase[0])),eks,-np.abs(hilbert(Ephase[0])),label='$\phi_s = 0$')
ax.plot(eks,np.abs(hilbert(Ephase[1])),eks,-np.abs(hilbert(Ephase[1])),label='$\phi_s = \pi/4$')
ax.plot(eks,np.abs(hilbert(Ephase[2])),eks,-np.abs(hilbert(Ephase[2])),label='$\phi_s = \pi/2$')
ax.plot(eks,np.abs(hilbert(Ephase[3])),eks,-np.abs(hilbert(Ephase[3])),label='$\phi_s = \\frac{3 \pi}{4}$')
ax.legend(loc='lower right')
ax.set_xlabel('$x/ \lambda_d$')

fig.tight_layout()
plt.show()
#fig.savefig('/home/christoffer/Thesis_MCs_figures/vacuumprop.pdf', format='pdf', bbox_inches='tight')

#Compare ne/gamme profiles!#
s=happi.Open('/home/christoffer/Downloads/Smilei/1D_LWDA_HPC/Output_june3_1000res')
ekspic = s.ParticleBinning(0).getAxis('x')/(2*np.pi)
ngampic = s.ParticleBinning(0).getData()

ekscold = ksigrid/(2*np.pi)
ngamcold = np.flip(WKF.wake_original(parameters,y0)[0])

fig, axs = plt.subplots(1, 2, figsize=set_size(subplots=(1.,2.)))
subtxt = ['(a)','(b)']

axs[0].plot(ekspic,ngampic[310],'-k')
axs[0].set_xlabel('$x / \lambda_d$')
axs[0].set_ylabel('$(n_e / n^{crit}_d)/\gamma_e$')
axs[0].set_xlim(11,43)
axs[0].text(0.03,0.9,subtxt[0],transform=axs[0].transAxes)


axs[1].plot(ekscold,ngamcold,'k')
axs[1].set_xlabel('$\\xi / \lambda_d$')
axs[1].set_ylabel('$(n_e / n^{crit}_d)/\gamma_e$')
axs[1].set_xlim(6.8,40)
axs[1].text(0.03,0.9,subtxt[1],transform=axs[1].transAxes)

fig.tight_layout()
plt.show()



##Multiple+Single mega comparison##
C = lambda t,tc,w: np.exp(-((t-tc)**10)/(w**10))
tcsing=131.38
wsing=4.
tcmult=131.38
wmult=4.
Ecomp = [Esing,Emult]
Ksing = np.fft.rfftfreq(n=len(Esing),d=1./resx)
Kmult = np.fft.rfftfreq(n=len(Emult),d=1./resx)
Signal = lambda(arr): np.abs(np.fft.rfft(arr))**2

kd = 1.
eks=Xgrid/(2*np.pi)
fig, axs = plt.subplots(2,2, figsize=set_size(subplots=(2.,2.)))
subtxt=['(a)','(b)','(c)','(d)']

#Mult field#
axs[0][0].set_xlabel('$x/\lambda_d$')
axs[0][0].set_ylabel('$E_s/E^0_s$')
axs[0][0].plot(eks,Ecomp[1])
axs[0][0].plot(eks,C(eks,tcmult,wmult)*np.amax(Ecomp[1]),'--k',label='Super-Gaussian filter')
axs[0][0].set_xlim(114,155)
axs[0][0].legend(loc='lower right')
axs[0][0].text(0.03,0.85,subtxt[0],transform=axs[0][0].transAxes)
#Sing field#
axs[1][0].set_xlabel('$x/\lambda_d$')
axs[1][0].set_ylabel('$E_s/E^0_s$')
axs[1][0].plot(eks,Ecomp[0])
axs[1][0].plot(eks,C(eks,tcsing,wsing)*np.amax(Ecomp[0]),'--k',label='Super-Gaussian filter')
axs[1][0].set_xlim(114,155)
axs[1][0].legend(loc='lower right')
axs[1][0].text(0.03,0.85,subtxt[2],transform=axs[1][0].transAxes)
#Mult power spec#
axs[0][1].set_xlabel('$k / k_d$')
axs[0][1].set_ylabel('$I(\omega) \ (a.u)$')
axs[0][1].plot(Kmult/kd,Signal(C(eks,tcmult,wmult)*Ecomp[1]),'c')
axs[0][1].set_xlim(0,5)
axs[0][1].text(0.03,0.85,subtxt[1],transform=axs[0][1].transAxes)
#Sing power spec#
axs[1][1].set_xlabel('$k / k_d$')
axs[1][1].set_ylabel('$I(\omega) \ (a.u)$')
axs[1][1].plot(Ksing/kd,Signal(C(eks,tcsing,wsing)*Ecomp[0]),'c')
axs[1][1].set_xlim(0,5)
axs[1][1].text(0.03,0.85,subtxt[3],transform=axs[1][1].transAxes)

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



