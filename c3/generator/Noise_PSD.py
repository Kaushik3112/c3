#!/usr/bin/env python
# coding: utf-8

# Importing the basic modules necessary for the code

# In[ ]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# At times when the noise PSD is given only for positive frequencies, in order to find the noise we take the noise PSD to be symmetric about 0.

# In[ ]:


def noise_spectrum(freqs):
    """ 
    In case the Noise PSD is not given, but you would like to simulate it with some known function, we can use this code to 
    simulate the noise frequency
    
    The exponential noise PSD is just an example and this should be changed according one's liking while using the code
    """
    
    freq_bin_max = int(len(freqs))-1
    S_lambda = np.zeros(len(freqs))
    for i in range(freq_bin_max//2):
        S_lambda[freq_bin_max-i] = 1/freqs[freq_bin_max-i]
        S_lambda[i] = S_lambda[freq_bin_max-i]
        
    return S_lambda 


# In[ ]:


def PSD_gen(PSD,PSDfreqs_neg = False):
    """
    If the PSD is known, we can use this function to extend the spectrum of the PSD to the negative side. This function does
    nothing in case the spectrum is defined on both sides of the origin.
    """
    
    if PSDfreqs_neg:
        return PSD
    else:
        length = len(PSD)
        PSD_new = np.zeros(2*length)
        for i in range(length):
            PSD_new[length-i-1] = PSD[length-i-1]
            PSD_new[i] = PSD[length-i]
        return PSD_new
    
    


# In order to extract the noise signal from the PSD, we must modify the PSD and then be operated on with the fourier tarnsform.
# The given function performs all the necessary operations in order to get the signal.

# In[ ]:


def Sig_gen(PSD):
    """
    PSD must be given in the order, 
    
                negative frequencies --> zero --> positive frequencies
    i.e. the order on a number line.(note that it is different from the order that numpy works in)
    """
    
    A = np.sqrt(PSD)
    K = len(A)//2
    nu = np.sqrt(2*np.pi)*np.random.rand(K)
    nud = -1*np.flip(nu)
    if len(A)//2 == len(A)/2:
        nul = np.concatenate((nud,nu), axis = None)
    else:
        nul = np.concatenate((nud,np.sqrt(2*np.pi)*np.random.rand(1)),axis =None)
        nul = np.concatenate((nul,nu), axis =None)
    Z = A*nul
    neg,pos = np.split(Z,[len(Z)//2])
    Zmod = np.concatenate((pos,neg),axis =None)
    beta = np.fft.ifft(Zmod)
    
    return beta


# The PSD of a signal is the fourier transform of the auto correlation function, which is nothing but the convolution of the signal with itself. In order to do this, we use the properties of fourier transform (makes it a lot simpler) and find the PSD of the signal by first performing the fourier transform directly and then working with it. This in general gives us a PSD with a scaling factor, we therefore normalize it to the range 0-1. 

# In[ ]:


def PSDcalculator(Signal):
    
    signalfft = np.fft.fft(Signal)
    freqs = np.fft.fftfreq(Signal.shape[-1])
    sigpos, signeg =  np.split(signalfft,[len(signalfft)//2+1])
    signalfft = np.concatenate((signeg,sigpos),axis = None)
    freqspos,freqsneg = np.split(freqs,[len(freqs)//2+1])
    freqs = np.concatenate((freqsneg,freqspos),axis = None)
    signalfft2 = np.flip(signalfft)
    PSD = signalfft*signalfft2
    PSD = abs(PSD)
    PSD = PSD/PSD[PSD.argmax()]
    return PSD,freqs


# ### Tests
# This is done to ensure that the loop is closed and does not produce any unncessary errors.

# In[ ]:


# Power series is given
# w = np.linspace(-100,100,100001)
# F = noise_spectrum(w)

# # A = np.sqrt(F)
# # Z = Z_gen(A)
# # beta = invfour(Z)
# beta = Sig_gen(F)


# # In[ ]:


# PSD3, freqs = PSDcalculator(beta)

# print(freqs)


# # In[ ]:


# plt.plot(freqs,PSD3)
# plt.plot(freqs,noise_spectrum(w))

