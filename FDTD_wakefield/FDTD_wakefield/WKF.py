import numpy as np
from scipy.integrate import odeint
from scipy.signal import argrelextrema
l0 = 2*np.pi #Reference length in this simulation inspired from 1D simulations
t0=l0  #Reference time inspired from 1D simulations
resx=105. #Spatial resolution def 105
rest=120. #Temporal resolution def 120
dt=t0/rest #Temporal grid length
dx=l0/resx #Spatial grid length
ne=0.026
vb=0.974 #Phase velocity of plasma wave
gamme=1./np.sqrt(1-vb**2)
omegad=1.
vp=1.
kd=omegad/vp

#Wakefield parameters#
ksigrid=np.arange(-100.,300.,dx) #Grid to evaluate model on
kp=np.sqrt(ne) #Plasma wavenumber
a0=1.9        #Normalized driver amplitude [0.6,2.5] for thesis
ksi0 = 1./(kp/np.sqrt(2)) #Driver RMS width
phiD=0.
push=0.
parameters = [ne,vb,gamme,kd,ksigrid,kp,a0,ksi0,phiD,push] #paramters for odeint


#Initial conditions#
y1_0 = 0.0
y2_0 = 0.0
y0 = [y1_0,y2_0]



#Driver Laser
A = lambda KSI,kd,a0,ksi0,phiD : a0*np.sin(kd*KSI+phiD)*np.exp(-(KSI)**2 / ksi0**2)

def phi(y,ksi,parameters):
  y1,y2 = y
  ne,vb,gamme,kd,ksigrid,kp,a0,ksi0,phiD,push = parameters
  derivatives = [y2, ((kp**2)*gamme**2)*(vb*((1.-(1.+A(ksi,kd,a0,ksi0,phiD)**2)/(gamme*(1.+y1))**2))**(-1./2) -1.)]

  return derivatives


def wake_original(parameters,y0):
 ne,vb,gamme,kd,ksigrid,kp,a0,ksi0,phiD,push = parameters
 psoln = odeint(phi,y0,ksigrid,args=(parameters,))
 PHI = psoln[:,0]
 ESTAT = -psoln[:,1]
 n_n0 = (vb*gamme**2)*( (1 - (1+A(ksigrid,kd,a0,ksi0,phiD)**2)/(gamme*(1+PHI))**2 )**(-1./2) - vb)
 #uz = ((1+PHI)*gamme**2)*(vb - (1 - (1+A(ksigrid,kd,a0,ksi0,phiD)**2)/(gamme*(1+PHI))**2 )**(1./2))
 gam = ((1+PHI)*gamme**2)*(1 - vb*(1 - (1+A(ksigrid,kd,a0,ksi0,phiD)**2)/(gamme*(1+PHI))**2 )**(1./2))
 ngam=((n_n0*ne)/gam)
 nn1 = n_n0-1


 return np.array([nn1,ESTAT])


def wake_bck(parameters,y0,setting='multiple'):
 ne,vb,gamme,kd,ksigrid,kp,a0,ksi0,phiD,push = parameters
 PHI = odeint(phi,y0,ksigrid,args=(parameters,))[:,0]
 n_n0 = (vb*gamme**2)*( (1 - (1+A(ksigrid,kd,a0,ksi0,phiD)**2)/(gamme*(1+PHI))**2 )**(-1./2) - vb)
 #uz = ((1+PHI)*gamme**2)*(vb - (1 - (1+A(ksigrid,kd,a0,ksi0,phiD)**2)/(gamme*(1+PHI))**2 )**(1./2))
 gam = ((1+PHI)*gamme**2)*(1 - vb*(1 - (1+A(ksigrid,kd,a0,ksi0,phiD)**2)/(gamme*(1+PHI))**2 )**(1./2))
 ngam=((n_n0*ne)/gam)
 ksilist=ksigrid.tolist()

 if setting == 'multiple':
  idxs=argrelextrema(ngam,np.less,order=50)[0]
  ksipart=ksilist[0:idxs[0]]
  ksiwake=ksilist[idxs[0]:idxs[1]]
  N=(len(ksigrid)-(2*len(ksipart)))/(len(ksiwake))
  laserpart=ngam[0:idxs[0]].tolist()
  laserend=ngam[0:idxs[0]].tolist()
  laserend.reverse()
  onewake=ngam[idxs[0]:idxs[1]].tolist()
  fullwake=laserpart+N*onewake+laserend+[laserend[-1]]*(len(ksigrid)-2*len(ksipart)-N*len(ksiwake))
 
 if setting == 'single':
  idxs=argrelextrema(ngam,np.less,order=50)[0]
  ksipart=ksilist[0:idxs[0]]
  ksiwake=ksilist[idxs[0]:idxs[1]]
  N=1
  laserpart=ngam[0:idxs[0]].tolist()
  laserend=ngam[0:idxs[0]].tolist()
  laserend.reverse()
  onewake=ngam[idxs[0]:idxs[1]].tolist()
  fullwake=laserpart+N*onewake+laserend+[laserend[-1]]*(len(ksigrid)-2*len(ksipart)-N*len(ksiwake)) 
  
 
 return np.array(fullwake)



def wake_zero(parameters,y0,setting='multiple'):
 ne,vb,gamme,kd,ksigrid,kp,a0,ksi0,phiD,push = parameters
 PHI = odeint(phi,y0,ksigrid,args=(parameters,))[:,0]
 n_n0 = (vb*gamme**2)*( (1 - (1+A(ksigrid,kd,a0,ksi0,phiD)**2)/(gamme*(1+PHI))**2 )**(-1./2) - vb)
 #uz = ((1+PHI)*gamme**2)*(vb - (1 - (1+A(ksigrid,kd,a0,ksi0,phiD)**2)/(gamme*(1+PHI))**2 )**(1./2))
 gam = ((1+PHI)*gamme**2)*(1 - vb*(1 - (1+A(ksigrid,kd,a0,ksi0,phiD)**2)/(gamme*(1+PHI))**2 )**(1./2))
 ngam=((n_n0*ne)/gam)
 ksilist=ksigrid.tolist()

 if setting == 'multiple':
  idxs=argrelextrema(ngam,np.less,order=50)[0]
  ksipart=ksilist[0:idxs[0]]
  ksiwake=ksilist[idxs[0]:idxs[1]]
  N=(len(ksigrid)-(2*len(ksipart)))/(len(ksiwake))
  laserpart=(ngam[idxs[0]]*np.ones(len(ksipart))).tolist()
  onewake=ngam[idxs[0]:idxs[1]].tolist()
  fullwake=laserpart+N*onewake+laserpart+[laserpart[-1]]*(len(ksigrid)-2*len(ksipart)-N*len(ksiwake)) 
 
 if setting == 'single':
  idxs=argrelextrema(ngam,np.less,order=50)[0]
  ksipart=ksilist[0:idxs[0]]
  ksiwake=ksilist[idxs[0]:idxs[1]]
  N=1
  laserpart=(ngam[idxs[0]]*np.ones(len(ksipart))).tolist()
  onewake=ngam[idxs[0]:idxs[1]].tolist()
  fullwake=laserpart+N*onewake+laserpart+[laserpart[-1]]*(len(ksigrid)-2*len(ksipart)-N*len(ksiwake)) 
  
 
 return np.array(fullwake)














