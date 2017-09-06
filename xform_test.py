#!/bin/python
# Code written by Robert A. Maxwell
# for coursework in DSP completed at
# University of Texas at San Antonio
# under instruction from
# Dr. Sos Agaian (02/26/2015)
# --Initially written for matlab and
#   converted to Python (Sept 06, 2017)

import numpy as np
import matplotlib.pyplot as plt

from freq_funcs import myDFT
from freq_funcs import dctTwo
from freq_funcs import myFFT, evenOddRecursive

from scipy.fftpack import dct, fft
from scipy.linalg import dft
import timeit
#
#

Fs = 1.0        #sample freq is discrete, not fractional
                #     freq > 1 has undesired effects
                #     freq < 1 can create signal aliasing
                #TODO: Should be able to have sampling freq
                #      for sufficiently large Fs (small step size)
                
t = 64          #num points/ signal length
nn = np.arange(start=0, stop=t, step=1/Fs)

####Periodic signal
period = 16         #period should be even and
                    # a factor of t (half,fourth, etc...)
omega = 2.0*np.pi/period
x = np.cos(omega*nn)
####Aperiodic signal
#x = ((0.9)**nn)*np.cos(0.1*np.pi*nn) 

k = np.arange(start=0, stop=t, step=1/Fs)   

plt.figure(1)
plt.subplot(111)
plt.plot(k,x)
plt.xlabel('n - time units')
plt.ylabel('x[n]')
plt.title('x[n]')
plt.grid(True)
plt.show()

##################################################
#Discrete Fourier Transform
##################################################

print '************************************'

start_time = timeit.default_timer()
X = myDFT(x,t,Fs)
elapsed = timeit.default_timer() - start_time
print 'myDFT time:   \t', elapsed

plt.figure(2)
plt.subplot(211)
markerline,stemlines,baseline = plt.stem(k,X,'-')
plt.xlabel('k')
plt.ylabel('|X[k]|')
plt.title('DFT -- x[n]') 

###         Sanity Check
start_time = timeit.default_timer()
mat = dft(Fs*t)
X = mat.dot(x).real
elapsed = timeit.default_timer() - start_time
print 'Scipy FFT time: ', elapsed

plt.subplot(212)
markerline,stemlines,baseline = plt.stem(k,X.real,'-')
plt.xlabel('k')
plt.ylabel('|X[k]|')
plt.title('DFT-Scipy.linalg-Check  --  x[n]') 
plt.tight_layout()
plt.show()

###################################################
# Discrete Cosine Transform
###################################################
print '************************************'

start_time = timeit.default_timer()
X = dctTwo(x,t,Fs)
elapsed = timeit.default_timer() - start_time
print 'myDCT time:   \t', elapsed

plt.figure(3)
plt.subplot(211)
markerline,stemlines,baseline = plt.stem(k,X,'-')
plt.xlabel('k')
plt.ylabel('|X[k]|')
plt.title('DCT-2  --  x[n]') 

###         Sanity Check
start_time = timeit.default_timer()
X = dct(x, type=2, n=Fs*t, norm='ortho')
elapsed = timeit.default_timer() - start_time
print 'Scipy DCT time: ', elapsed

plt.subplot(212)
markerline,stemlines,baseline = plt.stem(k,X,'-')
plt.xlabel('k')
plt.ylabel('|X[k]|')
plt.title('DCT-Scipy-Check  --  x[n]') 
plt.tight_layout()
plt.show()
###################################################

print '************************************'

start_time = timeit.default_timer()
X=myFFT(x,t)
X = evenOddRecursive(X,t)
elapsed = timeit.default_timer() - start_time
print 'myFFT time:   \t', elapsed

plt.figure(4)
plt.subplot(211)
markerline,stemlines,baseline = plt.stem(k,X.real,'-')
plt.xlabel('k - time units')
plt.ylabel('|X[k]|')
plt.title('FFT  --  x[n]')

###         Sanity Check
start_time = timeit.default_timer()
X = fft(x)
elapsed = timeit.default_timer() - start_time
print 'Scipy FFT time: ', elapsed

plt.subplot(212)
markerline,stemlines,baseline = plt.stem(k,X.real,'-')
plt.xlabel('k')
plt.ylabel('|X[k]|')
plt.title('FFT-Scipy-Check  --  x[n]') 
plt.tight_layout()


plt.grid(True)
plt.show()
