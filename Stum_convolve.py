# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 18:09:51 2012

@author: alst7468
"""

import numpy as np

def CreateK(RxC=(3,3),shape='rect',stat='mean'):
    '''This function returns a kernel as a numpy array.
    Parameters:
    Name:       Type:       Description:
    RxC         tuple       with one or two ints.
                            If a tuple with one int, assumed symmetrical focal window
                            Circle radius defined by the diagnal to corner of rectangle
    shape       string      rect for rectangular extent, circ for circle extent
    stat        string      To define focal statistic, mean or sum
    '''
    if shape == 'rect':
        if RxC.__len__() ==1: RxC= RxC*2 #create symmetrical window
        focal=np.ones(RxC,dtype=np.int8)
    elif shape == 'circ':
        rad=((RxC[0]/2.)**2+(RxC[1]/2.)**2)**.5    
        r=int(rad)    
        d=r*2+1
        mx=np.zeros((d,d))+np.arange(-r,r+1)
        my=mx.transpose()*-1
        focal=np.where((mx**2+my**2<=(rad)**2),1,0)*1
        RxC=(d,d)
    if stat=='mean':
        kernel = np.ones(RxC,dtype=np.float32)*(1.0/focal.sum())*focal
    elif stat=='sum':
        kernel = np.ones(RxC,dtype=np.float32)*focal
    return kernel
        

# Focal Mean function

def convolve(source,kernel):
    '''This function applies a kernel to a numpy array
    Parameters:
    Name:       Type:        Description:
    source      numpy array  Can be a 2-d or 3-d array
    kernel      numpy array  2-d numpy array with values to be applied at each 
                             convulotion step
    '''
    cc = np.where(kernel)
    rXc = kernel.shape
    RxC = source.shape

    if RxC.__len__() ==2:
        if kernel[kernel!=0].std()==0:
            if source.dtype==bool: d=np.int8
            else: d=np.float32
            out=np.zeros((RxC[0]-rXc[0]+1,RxC[1]-rXc[1]+1),dtype=d)
            k = kernel.max()
            for i in xrange(len(cc[0])):
                yf=cc[0][i]-(rXc[0]-1) or None
                xf=cc[1][i]-(rXc[1]-1) or None
                out+=source[cc[0][i]:yf,cc[1][i]:xf]
            return out*k
        else:
            out=np.zeros((RxC[0]-rXc[0]+1,RxC[1]-rXc[1]+1),dtype=np.float32)
            k = kernel.flatten() 
            for i in xrange(len(cc[0])):
                yi= cc[0][i]
                yf=cc[0][i]-rXc[0]+RxC[0]+1
                
                xi = cc[1][i]
                xf=cc[1][i]-rXc[1]+RxC[1]+1
                
                out+=source[yi:yf,xi:xf]*k[i]
            return out
            
    #For layer stack approach        
    elif RxC.__len__() == 3:
        out=np.zeros((RxC[0],RxC[1]-rXc[0]+1,RxC[2]-rXc[1]+1))
        if kernel[kernel!=0].std()==0:
            for i in xrange(len(cc[0])):
                yf=cc[0][i]-(rXc[0]-1) or None
                xf=cc[1][i]-(rXc[1]-1) or None
                out+=source[:,cc[0][i]:yf,cc[1][i]:xf]
            return out*kernel.max()
        else:
            k = kernel.flatten() 
            for i in xrange(len(cc[0])):
                yf=cc[0][i]-(rXc[0]-1) or None
                xf=cc[1][i]-(rXc[1]-1) or None
                out+=source[:,cc[0][i]:yf,cc[1][i]:xf]*k[i]
            return out
        
        
        