import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#Input parameters#
l0 = 2*np.pi #Reference length in this simulation inspired from 1D simulations
t0=l0  #Reference time inspired from 1D simulations
resx=105. #Spatial resolution, see dt and dx below
rest=120. #Temporal resolution
dt=t0/rest #Temporal grid length
dx=l0/resx #Spatial grid length
start=0. #Spatial grid start
stop=200.*l0 #Spatial grid end
tstart = 0. #Temporal grid start
tstop = 38.5*t0 #Temporal grid stop, set so that the interaction time should be equal that of a SMILEI simulation
Xgrid = np.arange(start,stop,dx) #Spatial grid with resolution dx
Tgrid = np.arange(tstart,tstop,dt) #Temporal grid with resolution dt
vb=0.974 #Phase velocity of plasma wave
gamme=1./np.sqrt(1-vb**2) #Corresponding gamma factor (if needed)
ne=0.026 #Background electron plasma density


###Everything needed to solve Wakefield Model!##################
#Parameters for Wakefield Model#
ksigrid=np.arange(-100.,300.,dx) #Grid to evaluate model on, pretty arbitrary grid but does the job
kp=np.sqrt(ne) #Plasma wavenumber
a0=2.5         #Normalized driver amplitude
ksi0 = 1./(kp/np.sqrt(2)) #Driver RMS width, see definition of driver laser below
parameters = [kp,a0,ksi0] #parameters for the numerical integration of wakefield model
ksicent = 50.             #Center for driver pulse
phiD = 0.                 #Phase for driver if needed

#Driver Laser
A = lambda KSI,a0,ksi0,phi,CENT : a0*np.sin(KSI+phi)*np.exp(-(KSI-CENT)**2 / ksi0**2)

#Initial conditions for numerical integration#
y1_0 = 0.0
y2_0 = 0.0
y0 = [y1_0,y2_0]

#Define function that will solve for electrostatic potential phi (FULL MODEL)#
def phi(y,ksi,parameters):
  y1,y2 = y
  kpsq,a0,ksi0 = parameters
  derivatives = [y2, ((kp**2)*gamme**2)*(vb*((1-(1+A(ksi,parameters[1],parameters[2],phiD,ksicent)**2)/(gamme*(1+y1))**2))**(-1./2) -1)]

  return derivatives

#Integration#
psoln = odeint(phi,y0,ksigrid,args=(parameters,))

#Quantities following from solution of phi#
PHI = psoln[:,0] #Electrostatic potential
ESTAT = -psoln[:,1] #Electrostatic field

#Fluid quantities#
n_n0 = (vb*gamme**2)*( (1 - (1+A(ksigrid,parameters[1],parameters[2],phiD,ksicent)**2)/(gamme*(1+PHI))**2 )**(-1./2) - vb)

uz = ((1+PHI)*gamme**2)*(vb - (1 - (1+A(ksigrid,parameters[1],parameters[2],phiD,ksicent)**2)/(gamme*(1+PHI))**2 )**(1./2))

gam = ((1+PHI)*gamme**2)*(1 - vb*(1 - (1+A(ksigrid,parameters[1],parameters[2],phiD,ksicent)**2)/(gamme*(1+PHI))**2 )**(1./2))
#############################################


#####FDTD functions, parameters etc..########### (Note: Steps taken are very similar to BSc Thesis on THz generation, please see their github)
#Output from wakefield model interpolated to be placed inside FDTD simulation grid#
push = 435. #used to displace the wakes inside FDTD simulation grid, this is not generalized yet
def NGAM(XArr,t):
  Dnsty = np.interp(XArr+vb*t-push,ksigrid,n_n0)*ne #interpolated density array
  Gm = np.interp(XArr+vb*t-push,ksigrid,gam) #Interpolated gamma array
  return np.flip(Dnsty/Gm) #Returns source term which is flipped so that driver is propagating to the right


#Arrays that can be filled with values for each timestep
Eout=np.zeros([len(Tgrid),len(Xgrid)]) #List of arrays to store spatial field in all timesteps
Nout=np.zeros([len(Tgrid),len(Xgrid)]) #List to be filled with density values
Pout=np.zeros([len(Tgrid),len(Xgrid)]) #List to be filled with transverse momentum values
Jout=np.zeros([len(Tgrid),len(Xgrid)]) #List to be filled with total transverse source term values



#Field solvers######
def Esolv(E,B,J,dt,dx,X,T):
  E = E+(dt/dx)*(B-np.roll(B,1))-dt*P*NGAM(X,T) #Update value of electric field with source term given by NGAM
  return E

def Bsolv(E,B,dt,dx):
  B = B+(dt/dx)*(np.roll(E,-1)-E) #Update value of magnetic field
  return B

def Psolv(E,P,dt):
  P = P+dt*E #Update value of transverse current density
  return P
####################

#Create arrays with zeros for them to be used for injecting a seed laser pulse#
E=np.zeros(len(Xgrid)) #Array for electric field
B=np.zeros(len(Xgrid)) #Array for magnetic field
P=np.zeros(len(Xgrid)) #Array for transverse momentum

#Initialize seed laser pulse, this one is forwards propagating in time to the right!#

PULSELENGTH = int(len(Xgrid)*0.5) #Gaussian pulse length
PULSESTART = (int(len(Xgrid))-PULSELENGTH)/2 #Gaussian pulse start in the spatial grid
E0 = 1. #Field amplitude, chosen quite arbitrarly to represent unity
OMEGAPRIM=0.2 #Seed frequency in SMILEI units
t_0 = 10.*t0 #Duration of temporal envelope, set to contain a few cycles

phiS=0. #Seed phase if needed
El = np.zeros(PULSELENGTH) #Pulse initialization array
Bl = np.zeros(PULSELENGTH) #-||-
START = -PULSELENGTH/2  #Start of the Gaussian pulse
STOPP = PULSELENGTH/2 #End of the Gaussian pulse
tEl = dx*np.arange(START,STOPP,1) #Grid to initalize pulse
tBl = dx*np.arange(START,STOPP,1)-(dx-dt)/2 #-||-
for i in range(len(El)):
  El[i] = E0*np.sin(OMEGAPRIM*tEl[i-1]+phiS)*np.exp(-(tEl[i-1]**2/(2*t_0**2)))  
  Bl[i] = -E0*(np.sin(OMEGAPRIM*tBl[i]+phiS)*np.exp(-(tBl[i]**2/(2*t_0**2))))

E[PULSESTART:PULSESTART+PULSELENGTH] = El #Now E and B contains the seed pulse in the first timeframe, onward to the FDTD simulation!
B[PULSESTART:PULSESTART+PULSELENGTH] = Bl


#Simulate the electromagnetic interaction, solves in the order E->B->P#
for Q in range(len(Tgrid)):
  E = Esolv(E,B,P,dt,dx,Xgrid,Tgrid[Q])      
  E[0]=0 #Reflecting boundary condition
  B = Bsolv(E,B,dt,dx)
  B[len(Xgrid)-1]=0 #Reflecting boundary condition
  P = Psolv(E,P,dt)
  Eout[Q] = E                      #Store field values inside list
  #Nout[Q] = NGAM(Xgrid,Tgrid[Q])   #Denstiy/Gamma storage, a bit superfluous
  Pout[Q] = P                      #Transverse momentum storage
  Jout[Q] = P*NGAM(Xgrid,Tgrid[Q]) #Transverse current storage

