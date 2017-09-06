#!/bin/python
#
# Code written by Robert A. Maxwell
# for coursework in DSP completed at
# University of Texas at San Antonio
# under instruction from
# Dr. Sos Agaian (02/26/2015)
# --Initially written for matlab and
#   converted to Python (Sept 06, 2017)

import numpy as np
import math

#function [ Xk ] = dctTwo( xn,N )
def dctTwo(xn,t,Fs):

# Computes the Discrete Cosine Transform Type 2
# (Signal wrap around)
#
# Doesn't match perfectly with scipy sanity check
#      most likely due to normalization strategy (betak)
    n = np.arange(start=0, stop=t, step=1/Fs)
    k = np.arange(start=0, stop=t, step=1/Fs)

    n = np.reshape(n, (n.shape[0],1))
    k = np.reshape(k, (k.shape[0],1))
    x = np.reshape(xn, (n.shape[0],1))
 
    betak = np.append(1/np.sqrt(2),np.ones((1,n.shape[0]-1)))    
    betak = np.reshape(betak,(n.shape[0],1))
 
    n = 2*n + 1
    nk = n.T*k      #create matrix for storing all eqns in k
        
    cosmat = np.cos((np.pi * nk)/(2*Fs*t))  #cosine matrix
    
    betak = math.sqrt(2.0/(Fs*t)) * betak
    betak = betak * x
    Xk = betak.T.dot(cosmat)
    return Xk.T

#
#
#
#function [ X ] = myDFT( x,N )
def myDFT(x,t,Fs):

# Computes the Discrete Fourier Transform
# of a finite length sequence
#
# Code adapted from
# Book: Digital Signal Processing using Matlab
# Ingle and Proakis
 
    n = np.arange(start=0, stop=t, step=1/Fs)  #[0:1:N-1]
    k = np.arange(start=0, stop=t, step=1/Fs)
    
    n = np.reshape(n,(n.shape[0],1))
    k = np.reshape(k,(k.shape[0],1))
 
    WsubN = np.exp(-1j*2*np.pi/(Fs*t))
    
    nk = n.T*k                 #create matrix for storing 
                                #all equations in k 
    WNnk = WsubN ** nk          
    X = x.dot(WNnk)
    
    return X.real

#
# Recursive FFT
#

#function [ Xk ] = myFFT( xn,N)
def myFFT(xn,t):
#   Recursive FFT function for Decimation-in-Frequency
#
#
#
    half_t = t/2    #this calculation happens a lot
    Xk = 0
    
    n2 = np.arange(start=0,stop=half_t ,step=1)
    
    if t > 2:    
    
        g = xn[0:half_t:1] + xn[half_t:t:1]
        h = xn[0:half_t:1] - xn[half_t:t:1]
        WsubN = np.exp(-1j*2*np.pi/t)

        WNn = WsubN ** n2
        h = WNn * h
                
        Xk = myFFT(g,half_t)
        Xk = [Xk,myFFT(h,half_t)]        #might kick up error
    
    else:
        g = xn[0] + xn[1]
        h = xn[0] - xn[1]
        Xk = [g,h]

    return np.array(Xk).flatten()

 
#
#Even Odd Sorting Algorithm
#

#function [y] = evenOddRecursive(x,N)
def evenOddRecursive(x,t):
# Separates even and odd indexes of above myFFT
#     algorithm and orders them such that
#     they match proper FFT output
#
# x = vector to be split
# N = length of x
    y = 0
    if t > 2:
        xeven = x[0:t:2]
        xodd = x[1:t:2]
        y = evenOddRecursive(xeven,t/2)    
        y = [y,evenOddRecursive(xodd,t/2)]
    
    else:
        xeven = x[0]
        xodd = x[1]
        y = [xeven,xodd]

    return np.array(y).flatten()
