import numpy as np
import matplotlib as mpl
#mpl.use('AGG')
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.signal import hilbert
import FuncDTD
import WKF

from my_plot import set_size
from scipy.stats import linregress
from scipy.signal import find_peaks

#plt.ion()

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

ne=0.0256 #ambient plasma density
omegad=1. #driver normalized frequency
vphi =  0.97#np.sqrt(1.-ne/omegad**2) #phase velocity of wake/group velocity of laser
gamme=1./np.sqrt(1-vphi**2) #Electron beam gamma factor
vp=1./vphi #Phase velocity of driver laser
kd=omegad/vp #Corresponding wavenumber for driver laser inside plasma

#Wakefield parameters#
ksigrid=np.arange(-100.,300.,dx) #Grid to evaluate model on
kp=np.sqrt(ne) #Plasma wavenumber
lambdap = 2.*np.pi/kp # Plasma wavelength
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
NgammaOut=np.zeros([len(Tgrid),len(Xgrid)]) 

#Create arrays with zeros for them to be filled with EM-values later#
E=np.zeros(len(Xgrid))
B=np.zeros(len(Xgrid))
P=np.zeros(len(Xgrid))


#Initialize seed laser pulse, this one is forwards propagating in time!#
PULSELENGTH = int(len(Xgrid)*0.4) #Gaussian pulse length
PULSESTART = int(len(Xgrid)*0.26) #Gaussian pulse start in the spatial grid
A0 = 1. #Field amplitude, chosen quite arbitrarly to represent unity
OMEGAPRIM=0.2 #2.*np.sqrt(ne) #Seed frequency in SMILEI units
E0 = A0*OMEGAPRIM
l_0 = 20.*t0 #Duration of temporal envelope, set to contain a few cycles
phiS=0. #Seed phase
KPRIM=np.sqrt(OMEGAPRIM**2-ne)
FuncDTD.GaussForward(E,B,A0,PULSELENGTH,PULSESTART,OMEGAPRIM,KPRIM,l_0,phiS,dt,dx) #Initialization of forwards propagating laser pulse
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
  NgammaOut[Q] = Ngamma
  
  Nprob[Q] = Ngamma[probs]	
  
  E = FuncDTD.Esolv(E,B,P,dt,dx,Ngamma)      
  E[0]=0
  B = FuncDTD.Bsolv(E,B,dt,dx)
  B[len(Xgrid)-1]=0
  P = FuncDTD.Psolv(E,P,dt)
  
  



# Process

#Electromagnetic field energy gain in time#
flt_wsg=20
STEP = 320 #*len(Tgrid)/NO #Number of steps to not sample too many energy data points
#qrange1 = range(0,1890,STEP)
qrange1 = [0,320,1900,2800,3150]
qrange2 = range(3800,20800,STEP)

qrange = qrange1 +qrange2

Egain = np.zeros(len(qrange)) #Array to store electromagnetic field values
EnvMaxTab = np.zeros(len(qrange)) #Array to store maximum of the envelope
EzMaxTab = np.zeros(len(qrange)) #Array to store maximum of Ez
KCentr = np.zeros(len(qrange)) #Array to store central k
lambCentr = np.zeros(len(qrange)) #Array to store central lambda
ifwhmTab = np.zeros(len(qrange)) #Array to store intensity fwhm duration 
ffwhmTab = np.zeros(len(qrange)) #Array to store field fwhm duration 
counter=0 #Just a hardcoded counter

xNorm = lambdap # Space is divided by xNorm for plotting

for q in qrange:
	Ez_anal = hilbert(Eout[q])
	Ez_env = np.abs(Ez_anal)
 	# global maximum
	#EzMaxX=Xgrid[pkindEz_env]
	#xCent = EzMaxX

	pkindNgamma, propertNg= find_peaks(NgammaOut[q],height=0.5*NgammaOut[q].max(),width=1)
	xNg = Xgrid[pkindNgamma[-1]]
	if q == 0:
		xNg0=xNg # record initial position of first density spike

	maskNg = np.exp(-np.power((Xgrid-xNg)/(0.75*lambdap),16))	
	ind_mx_Ez_env= np.argmax(maskNg*Ez_env)# Find maximum of envelope around first density spike only
	envMaxGlob = Ez_env[ind_mx_Ez_env]	

	# From last peak

	if q>qrange1[-1]:
	    pkindEz_env, propert0= find_peaks(maskNg*Ez_env,height=0.5*envMaxGlob,width=1)
	    pkindIz_env, propert1= find_peaks((maskNg*Ez_env)**2,height=(0.5*envMaxGlob)**2,width=1)
	elif q==0:
	    pkindEz_env, propert0= find_peaks(Ez_env,height=0.9*E0,width=1)
	    pkindIz_env, propert1= find_peaks((Ez_env)**2,height=(0.9*E0)**2,width=1)
	else:    
	    pkindEz_env, propert0= find_peaks(maskNg*Ez_env,height=1.1*E0,width=1)
	    pkindIz_env, propert1= find_peaks((maskNg*Ez_env)**2,height=(1.1*E0)**2,width=1)

	fwhm0=(propert0["right_ips"][-1]-propert0["left_ips"][-1])*dx
	fwhm1=(propert1["right_ips"][-1]-propert1["left_ips"][-1])*dx
	xCent=Xgrid[pkindEz_env[-1]]
	envMax = Ez_env[pkindEz_env[-1]]
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
		pkindEz_env=find_peaks(Ez_env,height=0.95*E0)
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
	
	pspE = pspect(maskNg*mask*Eout[q])
	
	idxKCentr = np.argmax(pspE)
	KCentr[counter] = K[idxKCentr]
	
	lamb=1./K[1:]
	
	pspElamb=pspE[1:]*1./lamb**2


	if counter<1: # ad hoc 
		idxlambCentr = np.argmax(pspElamb)
        elif q<3000:
                idxlambCentr=find_peaks(pspElamb)[0][1]
        else:
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
	
		plt.savefig('spectr_lambda_'+str(q)+'.png')
		plt.close()
	
	if 0:
		plt.figure()
		plt.plot((Xgrid-xNg0)/xNorm, Eout[q]/E0)
		plt.plot((Xgrid-xNg0)/xNorm, mask*Eout[q]/E0)
		plt.plot((Xgrid-xNg0)/xNorm, Ez_env/E0,'0.5')
		plt.plot((Xgrid-xNg0)/xNorm, -Ez_env/E0,'0.5')	
		plt.hlines(0.5*envMax/E0,(propert0["left_ips"][-1]*dx-xNg0)/xNorm, (propert0["right_ips"][-1]*dx-xNg0)/xNorm) # FWHM
		ax1=plt.twinx()
		plt.plot((Xgrid-xNg0)/xNorm, np.interp(Xgrid-vphi*Tgrid[q]-push,ksigrid,Profile),'r')
		
		plt.xlim((xNg-xNg0-0.5*lambdap)/xNorm,(xNg-xNg0+0.5*lambdap)/xNorm)

		plt.savefig('dens_fields_'+str(q)+'.png')
		#if q==0:
			#plt.show()
		#else:	
		plt.close()

		
	Egain[counter] = 0.5*np.trapz((Eout[q])**2+(Bout[q])**2,Xgrid) #Compute the electromagnetic energy
	counter+=1 
	
plt.ion()	
fig, ax = plt.subplots(1,1)
ax.set_ylabel('Field energy')
#ax.set_xlabel('$\omega_d t$')
ax.set_xlabel('$x/\lambda_d$') #$ \mathrm{(\mu m)}$')
ax.plot(vphi*Tgrid[qrange]/xNorm,Egain/Egain[0],':')
ax.plot(vphi*Tgrid[qrange]/xNorm,EnvMaxTab/E0,'--')
ax.plot(vphi*Tgrid[qrange]/xNorm,lambCentr,'.')
#ax2=plt.twinx()
#ax2.plot(vphi*qrange/xNorm,ffwhmTab/np.sqrt(2.),'r.')
ax.plot(vphi*Tgrid[qrange]/xNorm,ifwhmTab,'g-.')

plt.savefig('en_ampl_lamb_ifwhm.pdf')


