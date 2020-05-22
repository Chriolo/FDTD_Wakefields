import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.signal import hilbert
import FuncDTD
import neProf
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
tstop = 65.*t0 #Temporal grid stop, default is 45t0
Xgrid = np.arange(start,stop,dx) #Spatial grid with resolution dx
Tgrid = np.arange(tstart,tstop,dt) #Temporal grid with resolution dt
ne=0.026 #Background plasma density
vb = (1.-(3.*ne)/2.)  #Phase velocity of plasma wave
gamme = 1./np.sqrt(1-vb**2)

#Arrays
Eout=np.zeros([len(Tgrid),len(Xgrid)]) #List of arrays to store spatial field in all timesteps
Bout=np.zeros([len(Tgrid),len(Xgrid)]) #List to be filled with denisty values
Xwhere = np.zeros(len(Tgrid))

#Create arrays with zeros for them to be filled with EM-values#
E=np.zeros(len(Xgrid))
B=np.zeros(len(Xgrid))
P=np.zeros(len(Xgrid))



#Initialize seed laser pulse, this one is forwards propagating in time!#
PULSELENGTH = int(len(Xgrid)*0.4) #Gaussian pulse length
PULSESTART = int(len(Xgrid)*0.05) #Gaussian pulse start in the spatial grid
E0 = 1. #Field amplitude, chosen quite arbitrarly to represent unity
OMEGAPRIM=0.2 #Seed frequency in SMILEI units
t_0 = 10.*t0 #Duration of temporal envelope, set to contain a few cycles
phiS=0.
FuncDTD.GaussForward(E,B,E0,PULSELENGTH,PULSESTART,OMEGAPRIM,t_0,phiS,dt,dx)

#Density profile input#

#Box no ramp#
boxstart=Xgrid[PULSESTART+PULSELENGTH]
boxstop=Xgrid[PULSESTART+PULSELENGTH+int(0.1*len(Xgrid))]

#Box ramp#
STRT=Xgrid[PULSESTART+PULSELENGTH] #Start of box
DR=Xgrid[int(0.1*len(Xgrid))] #Box ramp up and down
DL=Xgrid[-1] #Box width

#Gaussian#
maxg=1.5
maxw=1.5
tgrow = np.linspace(1.,maxg,len(Tgrid)) #Fractional increase of amplitude
twid = np.linspace(1.,maxw,len(Tgrid)) #Fractional inrease/decrease of FWHM/width
amplitude = 0.18 #Amplitude i.e. maximum plasma density inspired from 1D simulations in terms of driver laser crit density
center=Xgrid[PULSESTART+int(PULSELENGTH/2.)] #Center of a gaussian profle.
te = 0.322*t0 #Initial 'Width' of Gaussian profile, taken from FWHM evolution data
offset=ne*0

#P = -integrate.cumtrapz(B,Xgrid,initial=0.) #Initialie transverse canonical momentum
#Simulate the electromagnetic interaction#
for Q in range(len(Tgrid)):
  E = FuncDTD.Esolv(E,B,P,dt,dx,neProf.box(Xgrid,1.,boxstart,boxstop))      
  E[0]=0
  B = FuncDTD.Bsolv(E,B,dt,dx)
  B[len(Xgrid)-1]=0
  P = FuncDTD.Psolv(E,P,dt)
  #Xwhere[Q] = Xgrid[np.where(np.abs(hilbert(E)) == np.amax(np.abs(hilbert(E))))[0][0]]
  Eout[Q] = E   #Store field values inside list

