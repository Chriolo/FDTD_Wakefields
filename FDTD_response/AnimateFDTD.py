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

#Plot time series of Electron beam#
eks=Xgrid/(2*np.pi)
fig, axs = plt.subplots(1,4, figsize=set_size(subplots=(1.,1.)))
subtxt=['(a)','(b)','(c)','(d)']
q = np.array(map(int,np.linspace(0,len(Tgrid)-1,4)))
qreal=Tgrid[q]/(2*np.pi)

for i in range(len(q)):
 axs[i].set_xlabel('$x/\lambda_d$')
 axs[i].set_ylabel('$E_s/E^0_s$')
 axs[i].plot(eks,Eout[q[i]])
 axs[i].plot(eks,20*neProf.gauss(Xgrid,Tgrid,vphi,Tgrid[q[i]],tgrow[q[i]],twid[q[i]],amplitude,center,te,offset,gamme,setting='normal'),'r',label='$n_e/\gamma_e$')
 axs[i].set_xlim(90+vphi*Tgrid[q[i]]/(2*np.pi),120+vphi*Tgrid[q[i]]/(2*np.pi)) 
 axs[i].set_ylim(-5,5)
 axs[i].set_title('$\omega_d t = %.2f$' %qreal[i])
 axs[i].legend(loc='lower right')
 axs[i].text(0.03,0.9,subtxt[i],transform=axs[i].transAxes)

fig.tight_layout()
plt.show()
#fig.savefig('/home/christoffer/Thesis_MCs_figures/plasmaprop.pdf', format='pdf', bbox_inches='tight')


#Plot CEP of electron beam in time#
fig, ax = plt.subplots(1,1)
time=Tgrid/(2*np.pi)

ax.set_ylabel('$E_s/E^0_s$')
ax.plot(time,Ephase[0,:],label='$\phi_s = 0$')
ax.plot(time,np.abs(hilbert(Ephase[0,:])),'k')
ax.plot(time,-np.abs(hilbert(Ephase[0,:])),'k')
ax.plot(time,Ephase[1,:],label='$\phi_s = \pi/4$')
ax.plot(time,Ephase[2,:],label='$\phi_s = \pi/2$')
ax.plot(time,Ephase[3,:],label='$\phi_s = \\frac{3 \pi}{4}$')
#ax.set_xlim(47,55)
ax.legend(loc='lower right')
ax.set_xlabel('$\omega_d t$')

plt.show()



