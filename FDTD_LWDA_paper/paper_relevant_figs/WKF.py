import numpy as np
from scipy.integrate import odeint
from scipy.signal import argrelextrema

#Driver Laser profile
A = lambda KSI,kd,a0,ksi0,phiD : a0*np.exp(-(KSI)**2 / ksi0**2)*np.sin(kd*KSI+phiD)

#odeint solver function
def phi(y,ksi,parameters):
  y1,y2 = y
  ne,vb,gamme,kd,ksigrid,kp,a0,ksi0,phiD,push = parameters
  #derivatives = [y2, ((kp**2)*gamme**2)*(vb*((1.-(1.+A(ksi,kd,a0,ksi0,phiD)**2)/(gamme*(1.+y1))**2))**(-1./2) -1.)]
  derivatives = [y2, (kp**2)*0.5*((1.+A(ksi,kd,a0,ksi0,phiD)**2)/(1+y1)**2-1.)]

  return derivatives

#Return density variation and electrostatic field for the thesis
def wake_original(parameters,y0):
 ne,vb,gamme,kd,ksigrid,kp,a0,ksi0,phiD,push = parameters
 psoln = odeint(phi,y0,ksigrid,args=(parameters,))
 PHI = psoln[:,0]
 #ESTAT = -psoln[:,1]
 #n_n0 = (vb*gamme**2)*( (1 - (1+A(ksigrid,kd,a0,ksi0,phiD)**2)/(gamme*(1+PHI))**2 )**(-1./2) - vb)
 #uz = ((1+PHI)*gamme**2)*(vb - (1 - (1+A(ksigrid,kd,a0,ksi0,phiD)**2)/(gamme*(1+PHI))**2 )**(1./2))
 #gam = ((1+PHI)*gamme**2)*(1 - vb*(1 - (1+A(ksigrid,kd,a0,ksi0,phiD)**2)/(gamme*(1+PHI))**2 )**(1./2))
 #ngam=((n_n0*ne)/gam)
 #nn1 = n_n0-1

 return np.array(ne/(1+PHI))

#Wake structure with background plasma
def wake_bck(parameters,y0,setting='multiple'):
 ne,vb,gamme,kd,ksigrid,kp,a0,ksi0,phiD,push = parameters
 PHI = odeint(phi,y0,ksigrid,args=(parameters,))[:,0]
 #n_n0 = (vb*gamme**2)*( (1 - (1+A(ksigrid,kd,a0,ksi0,phiD)**2)/(gamme*(1+PHI))**2 )**(-1./2) - vb)
 #uz = ((1+PHI)*gamme**2)*(vb - (1 - (1+A(ksigrid,kd,a0,ksi0,phiD)**2)/(gamme*(1+PHI))**2 )**(1./2))
 #gam = ((1+PHI)*gamme**2)*(1 - vb*(1 - (1+A(ksigrid,kd,a0,ksi0,phiD)**2)/(gamme*(1+PHI))**2 )**(1./2))
 ngam= ne/(1+PHI) # ((n_n0*ne)/gam)
 ksilist=ksigrid.tolist()

#Here, the background plasma is connected smoothly with the plasma wave oscillations according to the \xi-grid-length
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

#Same as above but here only a single wakefield oscillation is included
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


#Wake structure with cavity/zero background plasma
def wake_zero(parameters,y0,setting='multiple'):
 ne,vb,gamme,kd,ksigrid,kp,a0,ksi0,phiD,push = parameters
 PHI = odeint(phi,y0,ksigrid,args=(parameters,))[:,0]
 #n_n0 = (vb*gamme**2)*( (1 - (1+A(ksigrid,kd,a0,ksi0,phiD)**2)/(gamme*(1+PHI))**2 )**(-1./2) - vb)
 #uz = ((1+PHI)*gamme**2)*(vb - (1 - (1+A(ksigrid,kd,a0,ksi0,phiD)**2)/(gamme*(1+PHI))**2 )**(1./2))
 #gam = ((1+PHI)*gamme**2)*(1 - vb*(1 - (1+A(ksigrid,kd,a0,ksi0,phiD)**2)/(gamme*(1+PHI))**2 )**(1./2))
 ngam=ne/(1+PHI) #((n_n0*ne)/gam)
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














