import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from my_plot import set_size
from scipy.stats import linregress
from scipy.signal import hilbert

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

#Vacuum propagation#
q = np.array(map(int,np.linspace(0,len(Tgrid)-1,3)))
qreal=Tgrid[q]/(2*np.pi)
eks=Xgrid/(2*np.pi)
wy=Eout
fig, axs = plt.subplots(1, 3, figsize=set_size(subplots=(1.,1.)))
subtxt=['(a)','(b)','(c)']
# Plot
for i in range(len(q)):
 axs[i].axes.yaxis.set_visible(False)
 if i == 0:
  axs[i].set_ylabel('$E_s/E^0_s$')
  axs[i].axes.yaxis.set_visible(True)
 axs[i].plot(eks,wy[q[i]])
 axs[i].set_xlabel('$x/ \lambda_d$')
 axs[i].set_title('$\omega_d t = %.2f$' %qreal[i])
 axs[i].text(0.03,0.95,subtxt[i],transform=axs[i].transAxes)

plt.show()
#fig.savefig('/home/christoffer/Thesis_MCs_figures/vacuumprop.pdf', format='pdf', bbox_inches='tight')

#Plasma propagation#
q = np.array(map(int,np.linspace(0,len(Tgrid)-1,3)))
qreal=Tgrid[q]/(2*np.pi)
eks=Xgrid/(2*np.pi)
wy=Eout
den=neProf.boxramp(Xgrid,ne,STRT,DL,DR)
fig, axs = plt.subplots(1, 3, figsize=set_size(subplots=(1.,3.)))
# Plot
for i in range(len(q)):
 axs[i].axes.yaxis.set_visible(False)
 if i == 0:
  axs[i].set_ylabel('$E/E^0_s$')
  axs[i].axes.yaxis.set_visible(True)
 axs[i].plot(eks,wy[q[i]],eks,7*den,'r')
 axs[i].set_xlabel('$x/ \lambda_d$')
 axs[i].set_title('$t = %.2f/\omega_d$' %qreal[i])

plt.show()
#fig.savefig('/home/christoffer/Thesis_MCs_figures/plasmaprop.pdf', format='pdf', bbox_inches='tight')


#Plasma reflection#
q = np.array(map(int,np.linspace(0,len(Tgrid)-1,3)))
qreal=Tgrid[q]/(2*np.pi)
eks=Xgrid/(2*np.pi)
wy=Eout
den=neProf.box(Xgrid,1.,boxstart,boxstop)
fig, axs = plt.subplots(1, 3, figsize=set_size(subplots=(1.,1.)))
subtxt=['(a)','(b)','(c)']
# Plot
for i in range(len(q)):
 axs[i].axes.yaxis.set_visible(False)
 if i == 0:
  axs[i].set_ylabel('$E_s/E^0_s$')
  axs[i].axes.yaxis.set_visible(True)
 axs[i].plot(eks,wy[q[i]])
 axs[i].plot(eks,1.5*den,'r',label='$n_e$')
 axs[i].set_ylim(-2,2)
 axs[i].set_xlabel('$x/ \lambda_d$')
 axs[i].set_title('$\omega_d t = %.2f$' %qreal[i])
 axs[i].text(0.03,0.95,subtxt[i],transform=axs[i].transAxes)
 axs[i].legend(loc='lower right')


plt.show()
#fig.savefig('/home/christoffer/Thesis_MCs_figures/plasmaprop.pdf', format='pdf', bbox_inches='tight')


#Plasma propagation spectrum#
freqK = np.fft.rfftfreq(n=len(Eout[0]),d=1./resx)
Signal = lambda(arr): np.abs(np.fft.rfft(arr))**2
kp=np.sqrt(ne)
legtxt='lower right'

fig,ax = plt.subplots(figsize=set_size())
ax.plot(freqK/kp,Signal(Eout[0]),'b',label='Vacuum')
ax.plot(freqK/kp,Signal(Eout[-1]),'orange',label='Inside plasma')
ax.set_xlim(0,5)
ax.legend(loc=legtxt)
ax.set_xlabel('$k/k_p$')
ax.set_ylabel('I (a.u.)')

plt.show()
#fig.savefig('/home/christoffer/Thesis_MCs_figures/plasmaspec.pdf', format='pdf', bbox_inches='tight')

##Plasma group velocity##
#Plasma propagation#
eks=Xwhere/(2*np.pi)
wy=Tgrid/(2*np.pi)
fig, ax = plt.subplots(1, 1, figsize=set_size(subplots=(1.,1.)))
ax.plot(wy,eks)
ax.set_ylabel('$x/\lambda_d$')
ax.set_xlabel('$\omega_d t$')
vacslope=linregress(wy[0:4500],eks[0:4500])
plasslope=linregress(wy[8600:13100],eks[8600:13100])
print('Vacuum VG and stderr',vacslope[0],vacslope[-1],'and Plasma VG + stderr:',plasslope[0],plasslope[-1])

plt.show()
#fig.savefig('/home/christoffer/Thesis_MCs_figures/vg.pdf', format='pdf', bbox_inches='tight')

#Gaussian with and without offset#
eks=Xgrid/(2*np.pi) 
fig, axs = plt.subplots(1, 2, figsize=set_size(subplots=(1.,2.)))
Eut=[E0ne,Ene]
subtxt=['(a)','(b)']
for i in [0,1]:
 if i == 0:
  axs[i].set_ylabel('$E_s/E^0_s$')
 axs[i].plot(eks,Eut[i])
 axs[i].set_xlabel('$x/\lambda_d$')
 axs[i].set_xlim(86,92)
 axs[i].text(0.03,0.85,subtxt[i],transform=axs[i].transAxes)

fig.tight_layout()
plt.show()
#fig.savefig('/home/christoffer/Thesis_MCs_figures/vg.pdf', format='pdf', bbox_inches='tight')


#Gaussian propagation (NORMAL)#
q = np.array(map(int,np.linspace(0,len(Tgrid)-1,3)))
qreal=Tgrid[q]/(2*np.pi)
eks=Xgrid/(2*np.pi)
wy=Eout
fig, axs = plt.subplots(1, 3, figsize=set_size(subplots=(1.,1.)))
subtxt=['(a)','(b)','(c)']
# Plot
for i in range(len(q)):
 axs[i].axes.yaxis.set_visible(False)
 if i == 0:
  axs[i].set_ylabel('$E_s/E^0_s$')
  axs[i].axes.yaxis.set_visible(True)
 axs[i].plot(eks,wy[q[i]])
 axs[i].plot(eks,neProf.gauss(Xgrid,Tgrid,vphi,Tgrid[q[i]],tgrow[q[i]],twid[q[i]],amplitude,center,te,offset,setting='normal'),'r')
 axs[i].set_xlabel('$x/ \lambda_d$')
 axs[i].set_title('$\omega_d t = %.2f$' %qreal[i])
 axs[i].text(0.03,0.95,subtxt[i],transform=axs[i].transAxes)

plt.show()
#fig.savefig('/home/christoffer/Thesis_MCs_figures/vacuumprop.pdf', format='pdf', bbox_inches='tight')

#Gaussian ALL GROWTH COMPARISON#
eks=Xgrid/(2*np.pi)
Etot = [Enorm,Egrow,Ewid,Eboth]
fig, ax = plt.subplots(1, 1, figsize=set_size(subplots=(1.,1.)))

ax.set_ylabel('$E_s/E^0_s$')
ax.plot(eks,Etot[0],label='No growth')
ax.plot(eks,Etot[1],label='Amplitude growth')
ax.plot(eks,Etot[2],label='Width growth')
ax.set_xlim(88,89)
ax.plot(eks,Etot[3],label='Amplitude and width growth')
ax.legend(loc='lower right')
ax.set_xlabel('$x/ \lambda_d$')

fig.tight_layout()
plt.show()
#fig.savefig('/home/christoffer/Thesis_MCs_figures/vacuumprop.pdf', format='pdf', bbox_inches='tight')

#Gaussian CEP#
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

#Gaussian CEP HILBERT ENVELOPE#
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

##Gaussian power spec##
C = lambda t,tc,w: np.exp(-((t-tc)**10)/(w**10))
tc=88.
w=15.
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
axs[1].set_xlim(0,5)
axs[1].text(0.03,0.85,subtxt[1],transform=axs[1].transAxes)

fig.tight_layout()
plt.show()
#fig.savefig('/home/christoffer/Thesis_MCs_figures/plasmaprop.pdf', format='pdf', bbox_inches='tight')





#Animation#############
for m in range(0,len(Tgrid),50):
  plt.clf()
  plt.plot(Xgrid,Eout[m],'c')
  plt.pause(0.05)

for m in range(0,len(Tgrid),50):
  plt.clf()
  plt.plot(Xgrid,Eout[m],'c')
  plt.plot(Xgrid,8*neProf.boxramp(Xgrid,ne,STRT,DL,DR),'r')
  plt.pause(0.05)

for m in range(0,len(Tgrid),50):
  plt.clf()
  plt.xlim(250+vphi*Tgrid[m],300+vphi*Tgrid[m])
  plt.plot(Xgrid,Eout[m],'c')
  plt.plot(Xgrid,8*neProf.gauss(Xgrid,Tgrid,vphi,Tgrid[m],tgrow[m],twid[m],amplitude,center,te,offset,setting='amp'),'r')
  plt.pause(0.05)


