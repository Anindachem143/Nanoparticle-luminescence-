# Nanoparticle-luminescence-
We have fitted the time resolved luminescence spectra of Nanoparticles (non linear regression) with the help of tri and 4 exponential function using Python
import pandas as pd
import numpy as np

df = pd.read_csv("V2 532NM.CSV", sep = ',')
df

df1 = df.iloc[1033:4997,:]
df1

df1.reset_index(drop = True, inplace = True)
df1

cols = ['B']
df1[cols] = df1[df1[cols]>0][cols]
df1.dropna()
df1

df1 = df1[np.isfinite(df1['B'])]
df1

import matplotlib.pyplot as plt
plt.plot(df1.A, df1.B,'o')
plt.title("V2 lifetime")
plt.xlim(0,0.005)
plt.ylim(0,2.0)
plt.show()

from math import *
%matplotlib inline
import scipy.optimize
from lmfit import Model
from lmfit import minimize, Parameters, Parameter, report_fit

t = df1['A'].values
rt = df1['B'].values
noisy = rt + 0.001*np.random.normal(size=len(rt))

# define objective function: returns the array to be minimized
def fcn2min(params, t, noisy):
    c0 = params['c0'].value
    c1 = params['c1'].value
    c2 = params['c2'].value
    c3 = params['c3'].value
    c4 = params['c4'].value
    c5 = params['c5'].value
    c6 = params['c6'].value
    c7 = params['c7'].value
    model = c0 + (c1*np.exp(-(t-c2)/c3)) + (c4*np.exp(-(t-c2)/c5)) + (c6*np.exp(-(t-c2)/c7))
    return model - noisy

# create a set of Parameters
params = Parameters()
params.add('c0', value= 0.1, min=0 )
params.add('c1', value= 0.04, min=0)
params.add('c2', value= 0.00003, min=0)
params.add('c3', value= 0.00006, min=0)
params.add('c4', value= 0.2, min=0 )
params.add('c5', value= 0.0004, min=0)
params.add('c6', value= 0.8, min=0)
params.add('c7', value= 0.0008, min=0)

params['c2'].vary = False

# do fit, here with leastsq model
result = minimize(fcn2min, params, args=(t, noisy))
# calculate final result
final = noisy + result.residual
# write error report
report_fit(result.params)

# try to plot results
try:
    import pylab
    pylab.plot(t, noisy, 'ko')
    pylab.plot(t, final, 'r')
    pylab.xlim(0,0.005)
    pylab.xlabel('Time(ns)')
    pylab.ylabel('Fluorescence Intensity')
    pylab.show()
except:
    pass

lmfit_Rsquared = 1 - result.residual.var()/np.var(noisy)
print('Fit R-squared:', lmfit_Rsquared, '\n')
print(result.params)

print('Fit X^2: ', result.chisqr)
print('Fit reduced-X^2:', result.redchi)

plt.plot(t,result.residual,'r')
