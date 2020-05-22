import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.signal import hilbert
import FuncDTD
import WKF
#Input parameters#
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
a0=2.5        #Normalized driver amplitude [0.6,2.5] for thesis
ksi0 = 1./(kp/np.sqrt(2)) #Driver RMS width
phiD=0.
push=Xgrid[int(0.4*len(Xgrid))]+40.
parameters = [ne,vphi,gamme,kd,ksigrid,kp,a0,ksi0,phiD,push] #paramters for odeint


#Initial conditions#
y1_0 = 0.0
y2_0 = 0.0
y0 = [y1_0,y2_0]

#Arrays
Eout=np.zeros([len(Tgrid),len(Xgrid)]) #List of arrays to store spatial field in all timesteps
Bout=np.zeros([len(Tgrid),len(Xgrid)]) #List to be filled with denisty values
Pout=np.zeros([len(Tgrid),len(Xgrid)])

#Create arrays with zeros for them to be filled with EM-values#
E=np.zeros(len(Xgrid))
B=np.zeros(len(Xgrid))
#P=np.zeros(len(Xgrid))


#Initialize seed laser pulse, this one is forwards propagating in time!#
PULSELENGTH = int(len(Xgrid)*0.4) #Gaussian pulse length
PULSESTART = int(len(Xgrid)*0.3) #Gaussian pulse start in the spatial grid
E0 = 1. #Field amplitude, chosen quite arbitrarly to represent unity
OMEGAPRIM=0.2 #Seed frequency in SMILEI units
t_0 = 10.*t0 #Duration of temporal envelope, set to contain a few cycles
phiS=0.
FuncDTD.GaussForward(E,B,E0,PULSELENGTH,PULSESTART,OMEGAPRIM,t_0,phiS,dt,dx)
P = -integrate.cumtrapz(B,Xgrid,initial=0.)
Profile = WKF.wake_zero(parameters,y0)


#Simulate the electromagnetic interaction#
for Q in range(len(Tgrid)):
  E = FuncDTD.Esolv(E,B,P,dt,dx,np.interp(Xgrid-vphi*Tgrid[Q]-push,ksigrid,Profile))      
  E[0]=0
  B = FuncDTD.Bsolv(E,B,dt,dx)
  B[len(Xgrid)-1]=0
  P = FuncDTD.Psolv(E,P,dt)
  Eout[Q] = E   #Store field values inside list
  Bout[Q] = B

