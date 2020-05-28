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
tstop = 38.5*t0 #Temporal grid stop, default is 45t0
Xgrid = np.arange(start,stop,dx) #Spatial grid with resolution dx
Tgrid = np.arange(tstart,tstop,dt) #Temporal grid with resolution dt
ne=0.026  #Ambient plasma density
omegad=1. #Driver normalized frequency
vphi = np.sqrt(1.-ne/omegad**2) #Phase velocity of wake/Group velocity of driver laser
gamme = 1./np.sqrt(1-vphi**2) #Gamma factor of electron beam

#Arrays
Eout=np.zeros([len(Tgrid),len(Xgrid)]) #List of arrays to store spatial field in all timesteps
#Bout=np.zeros([len(Tgrid),len(Xgrid)]) #List to be filled with denisty values
#Xwhere = np.zeros(len(Tgrid))

#Create arrays with zeros for them to be filled with EM-values later#
E=np.zeros(len(Xgrid))
B=np.zeros(len(Xgrid))
P=np.zeros(len(Xgrid))

#Initialize seed laser pulse, this one is forwards propagating in time!#
PULSELENGTH = int(len(Xgrid)*0.4) #Gaussian pulse length
PULSESTART = int(len(Xgrid)*0.05) #Gaussian pulse start in the spatial grid
E0 = 1. #Field amplitude, chosen quite arbitrarly to represent unity
OMEGAPRIM=0.2 #Seed frequency in SMILEI units
t_0 = 10.*t0 #Duration of temporal envelope, set to contain a few cycles
phiS=0.    #Seed phase
FuncDTD.GaussForward(E,B,E0,PULSELENGTH,PULSESTART,OMEGAPRIM,t_0,phiS,dt,dx) #Initialization of forwards propagating laser pulse

#Density profile input#
#Box with no ramp#
boxstart=Xgrid[PULSESTART+PULSELENGTH] #Start of box plasma
boxstop=Xgrid[PULSESTART+PULSELENGTH+int(0.1*len(Xgrid))] #End of box plasma

#Box with ramp#
STRT=Xgrid[PULSESTART+PULSELENGTH] #Start of box plasma
DR=Xgrid[int(0.1*len(Xgrid))] #Box ramp up and down length
DL=Xgrid[-1] #Box flat top width

#Gaussian#
maxg=1.5 #Parameter used to determine how much the peak amplitude of the Gaussian will grow (increases by 50% in this case)
maxw=0.5 #Parameter used to determine how much the width of the Gaussian will decrease (down to 50% in this case)
tgrow = np.linspace(1.,maxg,len(Tgrid)) #Fractional increase of amplitude
twid = np.linspace(1.,maxw,len(Tgrid)) #Fractional decrease of FWHM/width
amplitude = 0.18 #Amplitude i.e. maximum plasma density inspired from 1D PIC simulations in terms of driver laser crit density
center=Xgrid[PULSESTART+int(PULSELENGTH/2.)] #Center of a gaussian profle.
te = 0.21*t0 #Initial 'Width' of Gaussian profile, taken from 1D PIC simulations
offset=ne*0.7 #Background cavity plasma, value taken from PIC simulations

P = -integrate.cumtrapz(B,Xgrid,initial=0.) #Initialize transverse canonical momentum
#Simulate the electromagnetic interaction#
for Q in range(len(Tgrid)):
  E = FuncDTD.Esolv(E,B,P,dt,dx,neProf.gauss(Xgrid,Tgrid,vphi,Tgrid[Q],tgrow[Q],twid[Q],amplitude,center,te,offset,setting='normal'))      
  E[0]=0
  B = FuncDTD.Bsolv(E,B,dt,dx)
  B[len(Xgrid)-1]=0
  P = FuncDTD.Psolv(E,P,dt)
  
  Eout[Q] = E   #Store field values inside list

