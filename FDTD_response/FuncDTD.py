import numpy as np

def Esolv(E,B,P,dt,dx,Nprof):
  E = E+(dt/dx)*(B-np.roll(B,1))+dt*P*Nprof #Update value of electric field, DEFALT B-np.roll(B,1) and vice versa for E
  return E

def Bsolv(E,B,dt,dx):
  B = B+(dt/dx)*(np.roll(E,-1)-E) #Update value of magnetic field
  return B

def Psolv(E,P,dt):
  P = P-dt*E #Update value of transverse current density
  return P

def GaussForward(E,B,E0,PULSELENGTH,PULSESTART,OMEGAPRIM, KPRIM,l_0,phiS,dt,dx):
 El = np.zeros(PULSELENGTH)
 Bl = np.zeros(PULSELENGTH)
 START = -PULSELENGTH/2  
 STOPP = PULSELENGTH/2
 xEl = dx*np.arange(START,STOPP,1)
 xBl = dx*np.arange(START,STOPP,1)+(dx-dt)/2
 for i in range(len(El)):
  El[i] = -OMEGAPRIM*E0*(np.sin(KPRIM*xEl[i]+phiS)*np.exp(-(xEl[i]**2/(2*l_0**2)))) + (KPRIM/OMEGAPRIM)*E0*(np.cos(KPRIM*xEl[i]+phiS)*np.exp(-(xEl[i]**2/(2*l_0**2))))*(-2*xEl[i]/(2*l_0**2))
  Bl[i] = KPRIM*E0*(np.sin(KPRIM*xBl[i]+phiS)*np.exp(-(xBl[i]**2/(2*l_0**2)))) - E0*(np.cos(KPRIM*xEl[i]+phiS)*np.exp(-(xEl[i]**2/(2*l_0**2))))*(-2*xEl[i]/(2*l_0**2))
 E[PULSESTART:PULSESTART+PULSELENGTH] = El
 B[PULSESTART:PULSESTART+PULSELENGTH] = Bl
