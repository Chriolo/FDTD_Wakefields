import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.signal import hilbert
import FuncDTD
import WKF

import matplotlib as mpl
import matplotlib.pyplot as plt
from my_plot import set_size
from scipy.stats import linregress
from scipy.signal import find_peaks

plt.ion()

#Input parameters#
l0 = 2*np.pi #Reference length in this simulation inspired from 1D simulations
t0=l0  #Reference time inspired from 1D simulations
resx=105. #Spatial resolution def 105
rest=120. #Temporal resolution def 120
dt=t0/rest #Temporal grid length
dx=l0/resx #Spatial grid length
start=0. #Spatial grid start
stop=350.*l0 #Spatial grid end, def 200 # 210
tstart = 0. #Temporal grid start, def 38
tstop = 200*t0 #Temporal grid stop, default is 38.5t0
Xgrid = np.arange(start,stop,dx) #Spatial grid with resolution dx
Tgrid = np.arange(tstart,tstop,dt) #Temporal grid with resolution dt

ne=0.026 #ambient plasma density
omegad=1. #driver normalized frequency
vphi = np.sqrt(1.-ne/omegad**2) #phase velocity of wake/group velocity of laser
gamme=1./np.sqrt(1-vphi**2) #Electron beam gamma factor
vp=1./vphi #Phase velocity of driver laser
kd=omegad/vp #Corresponding wavenumber for driver laser inside plasma

#Wakefield parameters#
ksigrid=np.arange(-100.,300.,dx) #Grid to evaluate model on
kp=np.sqrt(ne) #Plasma wavenumber
a0=2.5        #Normalized driver amplitude [0.6,2.5] for thesis
ksi0 = 1./(kp/np.sqrt(2)) #Driver RMS width
phiD=0.  #Driver phase
push=Xgrid[int(0.4*len(Xgrid))]-3. #Fine tune where the wakefield oscillations appear in the FDTD grid (Xgrid)
parameters = [ne,vphi,gamme,kd,ksigrid,kp,a0,ksi0,phiD,push] #paramters for odeint


#Initial conditions#
y1_0 = 0.0
y2_0 = 0.0
y0 = [y1_0,y2_0]

#Arrays
Eout=np.zeros([len(Tgrid),len(Xgrid)]) #List of arrays to store spatial field in all timesteps
Bout=np.zeros([len(Tgrid),len(Xgrid)]) 
#Pout=np.zeros([len(Tgrid),len(Xgrid)])

#Create arrays with zeros for them to be filled with EM-values later#
E=np.zeros(len(Xgrid))
B=np.zeros(len(Xgrid))
P=np.zeros(len(Xgrid))


#Initialize seed laser pulse, this one is forwards propagating in time!#
PULSELENGTH = int(len(Xgrid)*0.4) #Gaussian pulse length
PULSESTART = int(len(Xgrid)*0.26) #Gaussian pulse start in the spatial grid
E0 = 5. #Field amplitude, chosen quite arbitrarly to represent unity
OMEGAPRIM= 0.2 #Seed frequency in SMILEI units
l_0 = 20.*t0 #Duration of temporal envelope, set to contain a few cycles
phiS=0. #Seed phase
KPRIM=np.sqrt(OMEGAPRIM**2-ne)
FuncDTD.GaussForward(E,B,E0,PULSELENGTH,PULSESTART,OMEGAPRIM,KPRIM,l_0,phiS,dt,dx) #Initialization of forwards propagating laser pulse
P = -integrate.cumtrapz(B,Xgrid,initial=0.) #Initialie transverse canonical momentum
Profile = WKF.wake_bck(parameters,y0,setting='multiple') #Add the non-linear wakefield profile

#Probes
NO = 42 #Number of equidistant probes
probs = np.array(map(int,len(Xgrid)*(np.linspace(Xgrid[PULSESTART+PULSELENGTH/2],Xgrid[-1],NO)/(Xgrid[-1]))))-1 #Array of probe locations
Eprob = np.zeros([len(Tgrid),NO]) #Probe data array
Nprob = np.zeros([len(Tgrid),NO]) #Probe data array

#Simulate the electromagnetic interaction#
for Q in range(len(Tgrid)):
  Eout[Q] = E   #Store field values inside list
  Bout[Q] = B   
  Eprob[Q] = E[probs] #Store field values in probes
 
  Ngamma=np.interp(Xgrid-vphi*Tgrid[Q]-push,ksigrid,Profile)
  
  Nprob[Q] = Ngamma[probs]	
  
  E = FuncDTD.Esolv(E,B,P,dt,dx,Ngamma)      
  E[0]=0
  B = FuncDTD.Bsolv(E,B,dt,dx)
  B[len(Xgrid)-1]=0
  P = FuncDTD.Psolv(E,P,dt)
  



# Process

#Electromagnetic field energy gain in time#
flt_wsg=30
STEP = 4 #len(Tgrid)/NO #Number of steps to not sample too many energy data points
Egain = np.zeros(len(range(0,len(Tgrid),STEP))) #Array to store electromagnetic field values
EnvMaxTab = np.zeros(len(range(0,len(Tgrid),STEP))) #Array to store maximum of the envelope
EzMaxTab = np.zeros(len(range(0,len(Tgrid),STEP))) #Array to store maximum of Ez
KCentr = np.zeros(len(range(0,len(Tgrid),STEP))) #Array to store central k
lambCentr = np.zeros(len(range(0,len(Tgrid),STEP))) #Array to store central lambda
ifwhmTab = np.zeros(len(range(0,len(Tgrid),STEP))) #Array to store intensity fwhm duration 
ffwhmTab = np.zeros(len(range(0,len(Tgrid),STEP))) #Array to store field fwhm duration 
counter=0 #Just a hardcoded counter
for q in range(0,len(Tgrid),STEP):
	Ez_anal = hilbert(Eout[q])
	Ez_env = np.abs(Ez_anal)
	# From last peak
	pkindEz_env, propert0= find_peaks(Ez_env,height=0.9,width=1)
	fwhm0=(propert0["right_ips"][-1]-propert0["left_ips"][-1])*dx
	pkindIz_env, propert1= find_peaks(Ez_env**2,height=0.95,width=1)
	fwhm1=(propert1["right_ips"][-1]-propert1["left_ips"][-1])*dx
	xCent=Xgrid[pkindEz_env[-1]]
	envMax = Ez_env[pkindEz_env[-1]]
	# From global maximum
	#pkindEz_env= np.argmax(Ez_env)
	#EzMaxX=Xgrid[pkindEz_env]
	#xCent = EzMaxX
	#envMax = Ez_env[pkindEz_env]	
	#
	EnvMaxTab[counter]=envMax
	#pkindEz=find_peaks(Eout[q],height=0.3*envMax)
	#EzMax = Ez_env[pkindEz[0][-1]]
	#EzMaxTab[counter]=EzMax
	if 0:
		if counter<1: # ad hoc 
			mask = 1
		else:
			mask = np.exp(-np.power((Xgrid-xCent)/flt_wsg,16))	
		mask = np.exp(-np.power((Xgrid-xCent)/flt_wsg,16))	
		Ez_anal = hilbert(Eout[q]*mask)
		Ez_env = np.abs(Ez_anal)
		pkindEz_env=find_peaks(Ez_env,height=0.95)
		EzMaxX=Xgrid[pkindEz_env[0]]
		xCent = EzMaxX[-1]
		envMax = Ez_env[pkindEz_env[0][-1]]
		EnvMaxTab[counter]=envMax

	if counter<1: # ad hoc 
		mask = 1
	else:
		mask = np.exp(-np.power((Xgrid-xCent)/flt_wsg,16))	
	pspect = lambda(arr): np.abs(np.fft.rfft(arr))**2
	K = np.fft.rfftfreq(n=len(Eout[q]),d=1./resx)
	
	pspE = pspect(mask*Eout[q])
	
	idxKCentr = np.argmax(pspE)
	KCentr[counter] = K[idxKCentr]
	
	lamb=1./K[1:]
	
	pspElamb=pspE[1:]*1./lamb**2


	if counter<1: # ad hoc 
		idxlambCentr = np.argmax(pspElamb)
	else:   # ignore high wavelength peaks
		cutLamb = 5.
		ofs = pspElamb.shape[0]-pspElamb[lamb<=cutLamb].shape[0]
		idxlambCentr = ofs+np.argmax(pspElamb[lamb<=cutLamb])

	lambCentr[counter] = lamb[idxlambCentr]
	
	ffwhmTab[counter] = fwhm0/2./np.pi/lambCentr[counter]
	ifwhmTab[counter] = fwhm1/2./np.pi/lambCentr[counter]
	
	if 0:
		#plt.figure()
		
		#plt.plot(K,pspE)
		
		#plt.xlim(0,5)

		plt.figure()
		
		plt.plot(lamb,pspElamb)
		
		plt.xlim(0,10)
		
		plt.xlabel('lambda')
	
	if 0:
		plt.figure()
		plt.plot(Xgrid, Eout[q])
		plt.plot(Xgrid, mask*Eout[q])
		plt.plot(Xgrid, Ez_env,'0.5')
		plt.plot(Xgrid, -Ez_env,'0.5')	
		plt.hlines(0.5*envMax,propert0["left_ips"][-1]*dx, propert0["right_ips"][-1]*dx) # FWHM
		ax1=plt.twinx()
		plt.plot(Xgrid, np.interp(Xgrid-vphi*Tgrid[q]-push,ksigrid,Profile),'r')
	Egain[counter] = 0.5*np.trapz(Eout[q]**2+Bout[q]**2,Xgrid) #Compute the electromagnetic energy
	counter+=1 
fig, ax = plt.subplots(1,1)
ax.set_ylabel('Field energy')
#ax.set_xlabel('$\omega_d t$')
ax.set_xlabel('$x/\lambda_d$') #$ \mathrm{(\mu m)}$')
ax.plot(vphi*Tgrid[range(0,len(Tgrid),STEP)]/(2*np.pi),Egain/Egain[0],'.-')
#ax.plot(Tgrid[range(0,len(Tgrid),STEP)]/(2*np.pi),EnvMaxTab,'.')
ax.plot(vphi*Tgrid[range(0,len(Tgrid),STEP)]/(2*np.pi),lambCentr,'.-')
#ax2=plt.twinx()
#ax2.plot(Tgrid[range(0,len(Tgrid),STEP)]/(2*np.pi),ffwhmTab/np.sqrt(2.),'r.')
ax.plot(vphi*Tgrid[range(0,len(Tgrid),STEP)]/(2*np.pi),ifwhmTab,'g.-')
plt.show()


