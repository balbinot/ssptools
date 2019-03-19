#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as p
from scipy.interpolate import LinearNDInterpolator as interpnd

models = 'stdHe_afe2.all'

a,Z,Zeff,feh,afe,Y,m,logg,V,I,eep = np.loadtxt(models,
                                               usecols=(0,1,2,3,4,5,7,9,16,21,6),
                                               unpack=True)

feh = np.round(feh,1) ## To make life easier.
FEHS = np.unique(feh)
p.subplot(121)
for A in [7.0]:
    for B in np.unique(feh):
        print A, B
        jj = (feh==B)*(a==A)*(m>0.3)
        p.plot(V[jj]- I[jj], I[jj], '-', lw=2)
p.xlabel('V-I (fixed age)')
p.ylabel('V')
p.ylim(p.ylim()[::-1])

p.subplot(122)
for A in np.unique(a):
    for B in [FEHS[3]]:
        print A, B, 'fixfeh'
        jj = (feh==B)*(a==A)*(m>0.3)
        p.plot(V[jj]- I[jj], I[jj], '-', lw=2)
p.xlabel('V-I (fixed [Fe/H])')
p.ylabel('V')
p.ylim(p.ylim()[::-1])
p.show()


## Trim models for speed
j = (feh<=0)*(m>0.3)
N = len(a[j])
pts = np.zeros((N,3))
pts[:,0] = eep[j]
pts[:,1] = a[j]
pts[:,2] = feh[j]

jj = (a==5)*(feh==-1.5)
p.plot(V[jj]- I[jj], I[jj], '-', lw=2, label='age = 5 Gyr, feh=-1.5')
jj = (a==5)*(feh==-2.5)
p.plot(V[jj]- I[jj], I[jj], '-', lw=2, label='age = 5 Gyr, feh=-2.5')
jj = (a==5)*(feh==-2.0)
p.plot(V[jj]- I[jj], I[jj], '-', lw=2, label='age = 5 Gyr, feh=-2.0')

jj = (a==12)*(feh==-2.0)
p.plot(V[jj]- I[jj], I[jj], '-', lw=2, label='age = 12 Gyr, feh=-2.0')
jj = (a==12)*(feh==-2.5)
p.plot(V[jj]- I[jj], I[jj], '-', lw=2, label='age = 12 Gyr, feh=-2.5')
jj = (a==12)*(feh==-1.5)
p.plot(V[jj]- I[jj], I[jj], '-', lw=2, label='age = 12 Gyr, feh=-1.5')


#tri = np.load('asd.tri.npy')

iV = interpnd(pts, V[j])
iI = interpnd(iV.tri, I[j])

np.save('asd.tri', iV.tri)

FEH = -2.5
AGE = 5.0
ms = np.arange(1,300, 0.1) ## actually eep
for feh in [-1.75, -2.25]:
    p.plot(iV(ms, AGE,feh)-iI(ms, AGE,feh), iI(ms, AGE,feh), 'k:', label='age = {:.2f} Gyr, feh={:.2f}'.format(AGE, feh))
FEH = -2.5
AGE = 12.0
ms = np.arange(1,300, 0.1) ## actually eep
for feh in [-1.75, -2.25]:
    p.plot(iV(ms, AGE,feh)-iI(ms, AGE,feh), iI(ms, AGE,feh), 'k:', label='age = {:.2f} Gyr, feh={:.2f}'.format(AGE, feh))

p.xlabel('V-I')
p.ylabel('V')
p.legend(loc='best')
p.ylim(p.ylim()[::-1])
p.show()
exit()

