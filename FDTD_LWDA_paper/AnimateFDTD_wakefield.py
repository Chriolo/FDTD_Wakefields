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


###Plot Growth in time : region###
STRT=630. #Check interval of wake/field
STP=660.

fig, ax = plt.subplots(1,1)
ax.set_xlabel('$\omega_d t$')
ax.set_ylabel('$E_{max}/E^0_s$')
start=Xgrid[int(len(Xgrid)*(STRT/Xgrid[-1]))]
stop=Xgrid[int(len(Xgrid)*(STP/Xgrid[-1]))]
for T in range(len(Tgrid))[0::5]:
 first=start+vphi*Tgrid[T]
 last=stop+vphi*Tgrid[T]
 idxf = int(len(Xgrid)*(first/Xgrid[-1]))
 idxl = int(len(Xgrid)*(last/Xgrid[-1]))
 emax = np.amax(np.abs(hilbert(Eout[T][idxf:idxl])))
 ax.plot(Tgrid[T]/(2*np.pi),emax,'kx')
 ax.set_xlabel('$\omega_d t$')
 ax.set_ylabel('$E_{max}/E^0_s$')

plt.show()

###Plot freqmax in time : region###
from scipy.signal import find_peaks

fig, ax = plt.subplots(1,1)
ax.set_ylabel('$\omega/\omega_s$')
ax.set_xlabel('$x/\lambda_d$')
pwr = lambda(arr): np.abs(np.fft.rfft(arr))**2
C = lambda t,tc,w: np.exp(-((t-tc)**10)/(w**10))
w=12.

for P in range(NO):
 pks,locs = find_peaks(np.abs(hilbert(Eprob[:,P])),height=1.)
 idx=pks[0] #Location of the first field modulation in time
 cent=Tgrid[idx]
 signal = pwr(C(Tgrid,cent,w)*Eprob[:,P])
 freq = np.fft.rfftfreq(n=len(Eprob[:,P]),d=1./rest)
 KMAX = freq[np.where(signal == np.amax(signal))[0][0]]
 ax.plot(Xgrid[probs][P]/(2*np.pi),KMAX,'k.')

fig.tight_layout()
plt.show()










#Animation#############
for m in range(0,len(Tgrid),50):
  plt.clf()
  plt.plot(Xgrid,Eout[m],'c')
  plt.plot(Xgrid,8*np.interp(Xgrid-vphi*Tgrid[m]-push,ksigrid,Profile),'r')
  plt.pause(0.05)

#Range plot#
for m in range(0,len(Tgrid),50):
  plt.clf()
  plt.xlim(560+vphi*Tgrid[m],610+vphi*Tgrid[m])
  plt.plot(Xgrid,Eout[m],'c')
  plt.plot(Xgrid,8*np.interp(Xgrid-vphi*Tgrid[m]-push,ksigrid,Profile),'r')
  plt.pause(0.05)

#Range plot and envelope#
for m in range(0,len(Tgrid),50):
  plt.clf()
  plt.xlim(STRT+vphi*Tgrid[m],STP+vphi*Tgrid[m])
  plt.plot(Xgrid,Eout[m],'c')
  plt.plot(Xgrid,np.abs(hilbert(Eout[m])),'k',Xgrid,-np.abs(hilbert(Eout[m])),'k')
  plt.plot(Xgrid,8*np.interp(Xgrid-vphi*Tgrid[m]-push,ksigrid,Profile),'r')
  plt.pause(0.05)


for m in range(0,len(Tgrid),50):
  plt.clf()
  plt.plot(Xgrid,np.interp(Xgrid-vb*Tgrid[m],ksigrid,WKF.wake_bck(parameters,y0),'r'))
  plt.pause(0.05)



