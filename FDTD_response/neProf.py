import numpy as np

#Stationary box density distribution#
box = lambda x,n0,start,stop: n0*(np.heaviside(x-start,0.5)-np.heaviside(x-stop,0.5))

#Propagating box density distribution#
vbox =lambda x,t,v,n0,start,stop: n0*(np.heaviside(x-v*t-start,0.5)-np.heaviside(x-v*t-stop,0.5))

#Stationary box with linear ramp up and down#
boxramp = lambda x,n0,start,dl,dr: (n0/dr)*(x-start)*(np.heaviside(x-start,0.5)-np.heaviside(x-start-dr,0.5))+n0*(np.heaviside(x-start-dr,0.5)-np.heaviside(x-start-dr-dl,0.5))+(n0/dr)*(start+2*dr+dl-x)*(np.heaviside(x-start-dr-dl,0.5)-np.heaviside(x-start-2*dr-dl,0.5))

#Moving Gaussian density distribution#
def gauss(x,Tgrid,vb,t,tgrow,twid,amp,cent,sig,offset,gamme,setting='normal'): 

 if setting == 'normal':
  return (amp/gamme)*np.exp(-(x-cent-vb*t)**2 / (vb**2*sig**2))+(offset/gamme)

 if setting == 'amp':
  return (tgrow*(amp/gamme))*np.exp(-(x-cent-vb*t)**2 / (vb**2*sig**2))+(offset/gamme)

 if setting == 'wid':
  return (amp/gamme)*np.exp(-(x-cent-vb*t)**2 / (vb**2*twid*sig**2))+(offset/gamme)

 if setting == 'both':
  return (tgrow*(amp/gamme))*np.exp(-(x-cent-vb*t)**2 / (vb**2*twid*sig**2))+(offset/gamme)



