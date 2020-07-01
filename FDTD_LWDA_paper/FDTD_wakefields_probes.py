import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.signal import hilbert
import FuncDTD
import WKF
import neProf

#Input parameters#
l0 = 2*np.pi #Reference length in this simulation inspired from 1D simulations
t0=l0  #Reference time inspired from 1D simulations
resx=105. #Spatial resolution def 105
rest=120. #Temporal resolution def 120
dt=t0/rest #Temporal grid length
dx=l0/resx #Spatial grid length
start=0. #Spatial grid start
stop=210.*l0 #Spatial grid end, def 200
tstart = 0. #Temporal grid start, def 38
tstop = 40*t0 #Temporal grid stop, default is 38.5t0
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
PULSESTART = int(len(Xgrid)*0.3) #Gaussian pulse start in the spatial grid
E0 = 5. #Field amplitude, chosen quite arbitrarly to represent unity
OMEGAPRIM=0.2 #Seed frequency in SMILEI units
l_0 = 10.*t0 #Duration of temporal envelope, set to contain a few cycles
phiS=0. #Seed phase
KPRIM=np.sqrt(OMEGAPRIM**2-ne)
FuncDTD.GaussForward(E,B,E0,PULSELENGTH,PULSESTART,OMEGAPRIM,KPRIM,l_0,phiS,dt,dx) #Initialization of forwards propagating laser pulse
P = -integrate.cumtrapz(B,Xgrid,initial=0.) #Initialie transverse canonical momentum
Profile = WKF.wake_bck(parameters,y0,setting='multiple') #Add the non-linear wakefield profile
#Profile.fill(ne)
#Profile = neProf.boxramp(Xgrid,ne,Xgrid[PULSESTART+PULSELENGTH],60./kp,4./kp)

#Probes
NO = 12 #Number of equidistant probes
probs = np.array(map(int,len(Xgrid)*(np.linspace(Xgrid[PULSESTART+PULSELENGTH],Xgrid[-1],NO)/(Xgrid[-1]))))-1
Eprob = np.zeros([len(Tgrid),NO])


#Simulate the electromagnetic interaction#
for Q in range(len(Tgrid)):
  E = FuncDTD.Esolv(E,B,P,dt,dx,np.interp(Xgrid-vphi*Tgrid[Q]-push,ksigrid,Profile))      
  E[0]=0
  B = FuncDTD.Bsolv(E,B,dt,dx)
  B[len(Xgrid)-1]=0
  P = FuncDTD.Psolv(E,P,dt)
  Eout[Q] = E   #Store field values inside list
  Bout[Q] = B   #Store field values inside list

  Eprob[Q] = Eout[Q][probs]

