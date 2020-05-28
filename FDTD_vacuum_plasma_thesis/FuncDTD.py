import numpy as np

def Esolv(E,B,P,dt,dx,Nprof):
  E = E+(dt/dx)*(B-np.roll(B,1))-dt*P*Nprof #Update value of electric field
  return E

def Bsolv(E,B,dt,dx):
  B = B+(dt/dx)*(np.roll(E,-1)-E) #Update value of magnetic field
  return B

def Psolv(E,P,dt):
  P = P+dt*E #Update value of transverse momentum
  return P

#Forwards propagating laser pulse#
def GaussForward(E,B,E0,PULSELENGTH,PULSESTART,OMEGAPRIM,t_0,phiS,dt,dx):
 El = np.zeros(PULSELENGTH)
 Bl = np.zeros(PULSELENGTH)
 START = -PULSELENGTH/2  
 STOPP = PULSELENGTH/2
 tEl = dx*np.arange(START,STOPP,1)
 tBl = dx*np.arange(START,STOPP,1)+(dx-dt)/2
 for i in range(len(El)):
  El[i] = E0*(np.sin(OMEGAPRIM*tEl[i]+phiS)*np.exp(-(tEl[i]**2/(2*t_0**2))))
  Bl[i] = -E0*(np.sin(OMEGAPRIM*tBl[i]+phiS)*np.exp(-(tBl[i]**2/(2*t_0**2))))
 E[PULSESTART:PULSESTART+PULSELENGTH] = El
 B[PULSESTART:PULSESTART+PULSELENGTH] = Bl

