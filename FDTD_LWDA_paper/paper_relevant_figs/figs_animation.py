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


#Figure for plasma wake and subcycle+envelope#
fig, ax = plt.subplots(1,1)
Tcap=4500 #Arbitrary timestep to depict sub-cycle pulse generation
multiplier = 12 #Increase appearance of the ne/gammaE profile to depict its location in space
ax.set_xlabel('$x/\lambda_d$')
ax.set_ylabel('$E/E^0_s$')
ax.plot(Xgrid/(2*np.pi),Eout[Tcap]) #Field
ax.plot(Xgrid/(2*np.pi),np.abs(hilbert(Eout[Tcap])),'k') #Envelopes
ax.plot(Xgrid/(2*np.pi),-np.abs(hilbert(Eout[Tcap])),'k')
ax.plot(Xgrid/(2*np.pi),multiplier*np.interp(Xgrid-vphi*Tgrid[Tcap]-push,ksigrid,Profile),'r--',label='$n_e/\gamma_e$') #ne/gammaE profile
ax.set_xlim(140,148) #Limits to isolate first sub-cycle pulse
ax.legend(loc='lower right')
plt.show()




#Safecheck probe amplitude data from new placement#
from scipy.signal import find_peaks

for q in range(len(probs)):
 pks,locs = find_peaks(np.abs(hilbert(Eprob[:,q])),height=1.) #Find peaks and locations
 plt.clf() 
 plt.xlabel('$\omega_d t$')
 plt.ylabel('$E/E^0_s$')
 plt.plot(Tgrid/(2*np.pi),Eprob[:,q]) #Plot each time series for each probe
 plt.plot(Tgrid/(2*np.pi),np.abs(hilbert(Eprob[:,q])),'k') #Envelopes
 plt.plot(Tgrid[pks[0]]/(2*np.pi),np.abs(hilbert(Eprob[:,q]))[pks[0]],'rx')
 print(q)
 plt.pause(1)




#Amplitude growth plot#
RemovePts = 3 #number of points to remove (at the end) because at the end of the simulation the field is reflected and produce non-relevant data. Check for example the safecheck animation and pay attention towards the end.
mag = np.zeros(len(probs)-RemovePts) #Array to contain amplitude values
fig, ax = plt.subplots(1,1) 

for q in range(len(probs)-RemovePts):
 data=np.abs(hilbert(Eprob[:,q]))
 pks,locs = find_peaks(data,height=1.)
 mag[q] = data[pks[0]]
 print(data[pks[0]])

ax.set_xlabel('$x/\lambda_d$')
ax.set_ylabel('$max(E)/E^0_s$')
ax.plot(Xgrid[probs[0:(len(probs)-RemovePts)]]/(2*np.pi),mag,'k.')
plt.show()


#Safecheck probe data + corresponding filter#
from scipy.signal import find_peaks
pwr = lambda(arr): np.abs(np.fft.rfft(arr))**2 #Power spectrum function
C = lambda t,tc,w: np.exp(-((t-tc)**10)/(w**10)) #Super-Gaussian filter function
w=20. #Width of filter, quite arbitrary but a value around 20 will capture a pulse

for q in range(len(probs)):
 data=np.abs(hilbert(Eprob[:,q])) 
 pks,locs = find_peaks(data,height=1.)
 idx=pks[0] #Location of the first field modulation in time
 cent=Tgrid[idx] #This location will also be the center of the filter
 signal = pwr(C(Tgrid,cent,w)*Eprob[:,q]) #Compute spectral filter
 freq = np.fft.rfftfreq(n=len(Eprob[:,q]),d=1./rest) #Frequency array
 plt.clf()
 plt.xlabel('$\omega_d t$')
 plt.ylabel('$E/E^0_s$')
 plt.ylim((-9,9))
 plt.plot(Tgrid/(2*np.pi),Eprob[:,q],'b',Tgrid/(2*np.pi),8*C(Tgrid,cent,w),'k--')
 print(q)
 plt.pause(0.5)

#Safecheck probe filtered spectrum#
from scipy.signal import find_peaks
pwr = lambda(arr): np.abs(np.fft.rfft(arr))**2
C = lambda t,tc,w: np.exp(-((t-tc)**10)/(w**10))
w=12.

for q in range(len(probs)):
 data=np.abs(hilbert(Eprob[:,q]))
 pks,locs = find_peaks(data,height=1.)
 idx=pks[0] #Location of the first field modulation in time
 cent=Tgrid[idx]
 signal = pwr(C(Tgrid,cent,w)*Eprob[:,q])
 freq = np.fft.rfftfreq(n=len(Eprob[:,q]),d=1./rest)
 plt.clf()
 plt.xlim((0,12))
 plt.plot(freq/0.2,signal)
 print(q)
 plt.pause(1.5)



#Frequency growth plot#
RemovePts = 3 #number of points to remove (at the end) because at the end of the simulation the field is reflected and produce non-relevant data. Check for example the safecheck animation and pay attention towards the end.

omeg = np.zeros(len(probs)-RemovePts) #Array to store frequency values
fig, ax = plt.subplots(1,1)

for q in range(len(probs)-RemovePts):
 data=np.abs(hilbert(Eprob[:,q]))
 pks,locs = find_peaks(data,height=1.)
 idx=pks[0] #Location of the first field modulation in time
 cent=Tgrid[idx]
 signal = pwr(C(Tgrid,cent,w)*Eprob[:,q])
 freq = np.fft.rfftfreq(n=len(Eprob[:,q]),d=1./rest)
 omeg[q] = freq[np.where(signal == np.amax(signal))[0][0]]
 print(data[pks[0]])

ax.set_xlabel('$x/\lambda_d$')
ax.set_ylabel('$\omega/\omega_s$')
ax.plot(Xgrid[probs[0:(len(probs)-RemovePts)]]/(2*np.pi),omeg/0.2,'k.')
plt.show()



#Electromagnetic field energy gain in time#
STEP = len(Tgrid)/NO #Number of steps to not sample too many energy data points
Egain = np.zeros(len(range(0,len(Tgrid),STEP))) #Array to store electromagnetic field values
counter=0 #Just a hardcoded counter
for q in range(0,len(Tgrid),STEP):
 Egain[counter] = 0.5*np.trapz(Eout[q]**2+Bout[q]**2,Xgrid) #Compute the electromagnetic energy
 counter+=1
fig, ax = plt.subplots(1,1)
ax.set_ylabel('Field energy')
ax.set_xlabel('$\omega_d t$')
ax.plot(Tgrid[range(0,len(Tgrid),STEP)]/(2*np.pi),Egain)
plt.show()



#CEP from an arbitrary probe (can be chosen in FDTD_wakefields_probes_CEP)#
fig, ax = plt.subplots(1,1)
time=Tgrid/(2*np.pi)

ax.set_ylabel('$E_s/E^0_s$')
ax.plot(time,Ephase[0,:],label='$\phi_s = 0$')
ax.plot(time,np.abs(hilbert(Ephase[0,:])),'k')
ax.plot(time,-np.abs(hilbert(Ephase[0,:])),'k')
ax.plot(time,Ephase[1,:],label='$\phi_s = \pi/4$')
ax.plot(time,Ephase[2,:],label='$\phi_s = \pi/2$')
ax.plot(time,Ephase[3,:],label='$\phi_s = \\frac{3 \pi}{4}$')
ax.set_xlim(47,55)
ax.legend(loc='lower right')
ax.set_xlabel('$\omega_d t$')

plt.show()


#Paper combined plot!#
fig,ax = plt.subplots(1,1)
from scipy.signal import find_peaks
pwr = lambda(arr): np.abs(np.fft.rfft(arr))**2
C = lambda t,tc,w: np.exp(-((t-tc)**10)/(w**10))
w=20.

RemovePts = 3
RemoveStart = 6 #How many points to remove from the start
omeg = np.zeros(len(probs)-RemovePts)
for q in range(len(probs)-RemovePts):
 data=np.abs(hilbert(Eprob[:,q]))
 pks,locs = find_peaks(data,height=1.)
 idx=pks[0] #Location of the first field modulation in time
 cent=Tgrid[idx]
 signal = pwr(C(Tgrid,cent,w)*Eprob[:,q])
 freq = np.fft.rfftfreq(n=len(Eprob[:,q]),d=1./rest)
 omeg[q] = freq[np.where(signal == np.amax(signal))[0][0]]
 #print(data[pks[0]])

ax.set_xlabel('$x/\lambda_d$')
ax.set_ylabel('$\omega/\omega_s, \ max(E)/E^0_s, \ U/U^0_s$')
ax.plot(Xgrid[probs[RemoveStart:(len(probs)-RemovePts)]]/(2*np.pi),omeg[RemoveStart:]/0.2,'g.',label='$\omega/\omega_s$')

mag = np.zeros(len(probs)-RemovePts)

for q in range(len(probs)-RemovePts):
 data=np.abs(hilbert(Eprob[:,q]))
 pks,locs = find_peaks(data,height=1.)
 mag[q] = data[pks[0]]
 #print(data[pks[0]])

ax.plot(Xgrid[probs[RemoveStart:(len(probs)-RemovePts)]]/(2*np.pi),mag[RemoveStart:],'r.',label='$max(E)/E^0_s$')

STEP = len(Tgrid)/NO
RemovePtsEnergy = RemovePts+1
Egain = np.zeros(len(range(0,len(Tgrid),STEP)))
counter=0
Xspace = np.array((Xgrid[PULSESTART+PULSELENGTH/2]+vphi*Tgrid)/(2*np.pi))[range(0,len(Tgrid),STEP)]
for q in range(0,len(Tgrid),STEP):
 Egain[counter] = 0.5*np.trapz(Eout[q]**2+Bout[q]**2,Xgrid)
 counter+=1
ax.plot(Xspace[RemoveStart:len(Xspace)-RemovePtsEnergy],Egain[RemoveStart:len(Egain)-RemovePtsEnergy]/Egain[0],'b.',label='$U/U^0_s$')
ax.legend(loc='lower right')
plt.show()








#Animation#############
for m in range(0,len(Tgrid),50):
  plt.clf()
  plt.plot(Xgrid,Eout[m],'c')
  plt.plot(Xgrid,8*np.interp(Xgrid-vphi*Tgrid[m]-push,ksigrid,Profile),'r')
  plt.pause(0.05)


#Range plot and envelope#
STRT=650
STP=690
for m in range(0,len(Tgrid),50):
  plt.clf()
  plt.xlim(STRT+vphi*Tgrid[m],STP+vphi*Tgrid[m])
  plt.plot(Xgrid,Eout[m],'c')
  plt.plot(Xgrid,np.abs(hilbert(Eout[m])),'k',Xgrid,-np.abs(hilbert(Eout[m])),'k')
  plt.plot(Xgrid,8*np.interp(Xgrid-vphi*Tgrid[m]-push,ksigrid,Profile),'r')
  plt.pause(0.05)





