#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:14:14 2020

@author: vincentgousy-leblanc
"""

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import scipy
import pywt

def vector_position(theta,phi,psi):
    """Euler matrix"""
    a00=np.cos(psi)*np.cos(phi)-np.cos(theta)*np.sin(phi)*np.sin(psi)
    a01=np.cos(psi)*np.sin(phi)+np.cos(theta)*np.cos(phi)*np.sin(psi)
    a02=np.sin(psi)*np.sin(theta)
    
    a10=-np.sin(psi)*np.cos(phi)-np.cos(theta)*np.sin(phi)*np.cos(psi)
    a11=-np.sin(psi)*np.sin(phi)+np.cos(theta)*np.cos(phi)*np.cos(psi)
    a12=np.cos(psi)*np.sin(theta)
    
    a20=np.sin(theta)*np.sin(phi)
    a21=-np.sin(theta)*np.cos(phi)
    a22=np.cos(theta)
    
    euler_matrix=([a00,a01,a02],[a10,a11,a12],[a20,a21,a22])
    initial_vector=([0,0,1]) #z direction oriented
    new_vector=np.dot(euler_matrix,initial_vector)
    origin = [0], [0] # origin point
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(0,0,0,initial_vector[0],initial_vector[1],initial_vector[2],color="r",label='initial vector')
    ax.quiver(0,0,0,new_vector[0],new_vector[1],new_vector[2],color="c",label='final vector')
    #plt.quiver(*origin,initial_vector[1],initial_vector[2],scale=1,units="xy",color="r")
    #plt.quiver(*origin,new_vector[1],initial_vector[2],color="c",scale=1,units="xy")
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.legend()
    plt.show()
    return new_vector


def getPosition2(ax,ay,az,timen):
    """
    Calculate the position of the object (prediction)
    """
    #ax=ax*9.8
    #ay=ay*9.8
    az=(az-1)
    vx=np.zeros(len(ax))
    x=np.zeros(len(ax))
    x2=np.zeros(len(ax))
    vx2=np.zeros(len(ax))
    for i in range(0,len(ax)-1):
        vx[i+1]=vx[i]+ax[i+1]*(timen[i+1]-timen[i])
        #vx2[i]=np.trapz(np.array([ax[i-1],ax[i]]),np.array([timen[i-1],timen[i]]))
    for i in range(1,len(vx)-1):
        x[i]=x[i-1]+vx[i-1]*(timen[i]-timen[i-1])
        #x2[i]=np.trapz(np.array([vx2[i-1],vx2[i]]),np.array([timen[i-1],timen[i]]))
    plt.plot(timen,ax,'.')
    plt.show()
    plt.plot(timen,vx,'.')
    plt.xlabel('Time (sec)')
    plt.ylabel('speed (m/s)')
    plt.show()
    plt.plot(timen,x,'.')
    plt.xlabel('Time (sec)')
    plt.ylabel('Position (m)')
    plt.title("x")
    plt.show()
    vy=np.zeros(len(ay))
    y=np.zeros(len(ay))
    y2=np.zeros(len(ay))
    vy2=np.zeros(len(ay))
    for i in range(1,len(ay)):
        vy[i]=vy[i-1]+ay[i-1]*(timen[i]-timen[i-1])
        vy2[i]=np.trapz(np.array([ay[i-1],ay[i]]),np.array([timen[i-1],timen[i]]))
    for i in range(1,len(vy)):
        y[i]=y[i-1]+vy[i-1]*(timen[i]-timen[i-1])
        y2[i]=np.trapz(np.array([vy2[i-1],vy2[i]]),np.array([timen[i-1],timen[i]]))
    plt.title("y")
    plt.plot(timen,vy,'.')
    plt.xlabel('Time (sec)')
    plt.ylabel('speed (m/s)')
    plt.show()
    plt.title("y")
    plt.plot(timen,y,'.')
    plt.xlabel('Time (sec)')
    plt.ylabel('Position (m)')
    plt.show()
    #print(y)
    vz=np.zeros(len(az))
    z=np.zeros(len(az))
    z2=np.zeros(len(az))
    vz2=np.zeros(len(ay))
    for i in range(1,len(ay)):
        vz[i]=(abs(az[i]+az[i-1])/2)*(timen[i]-timen[i-1])
        vz2[i]=np.trapz(np.array([az[i-1],az[i]]),np.array([timen[i-1],timen[i]]))
    for i in range(1,len(vx)):
        z[i]=(abs(vz[i]+vz[i-1])/2)*(timen[i]-timen[i-1])
        z2[i]=np.trapz(np.array([vz2[i-1],vz2[i]]),np.array([timen[i-1],timen[i]]))
    plt.title("vz")
    plt.plot(timen,vz,'-')
    plt.xlabel('Time (sec)')
    plt.ylabel('speed (m/s)')
    plt.show()
    plt.title("z")
    plt.plot(timen,z,'.')
    plt.xlabel('Time (sec)')
    plt.ylabel('Position (m)')
    plt.show()
    #plt.plot(timen,ax,".-")
    #plt.show()
    #plt.plot(timen,ay,".-")
    #plt.show()
    #plt.plot(timen,az,".-")
    #plt.show()
    return x,y,z

def onAngleChange(acceleration):
    """
    Caculate the angle from the horizontal with the acceleration
    """
    gx=acceleration[0]
    gy=acceleration[1]
    gz=acceleration[2]
    
    tilt=np.arctan2(gx,gy)*(180/np.pi)
    
    eulertheta=np.arctan2(np.sqrt(gx**2+gy**2),gz)
    eulerphi=np.arctan2(gy,np.sqrt(gx**2+gz**2))
    eulerpsi=np.arctan2(gx,np.sqrt(gz**2+gy**2))
    
    print("Tilt 2D|theta|phi|psi: "+ str(tilt) + " | " + str(eulertheta) + " | " +str(eulerphi)+ " | " +str(eulerpsi))
    print("ax|ay|az: "+ str(gx) + " | " + str(gy) + " | " +str(gz))
    n=vector_position(eulertheta,eulerphi,eulerpsi)
    return n

def wavelet_denoise(data, wavelet, noise_sigma):
    '''Filter accelerometer data using wavelet denoising

    Modification of F. Blanco-Silva's code at: https://goo.gl/gOQwy5
    '''

    wavelet = pywt.Wavelet(wavelet)
    levels  = min(12,(np.floor(np.log2(np.shape(data)[0]))).astype(int))

    # Francisco's code used wavedec2 for image data
    wavelet_coeffs = pywt.wavedec(data, wavelet, level=levels)
    threshold = noise_sigma*np.sqrt(2*np.log2(np.size(data)))

    new_wavelet_coeffs = map(lambda x: pywt.threshold(x, threshold, mode='soft'),
                             wavelet_coeffs)

    return pywt.waverec(list(new_wavelet_coeffs), wavelet)


def rolling_average(x,N):
    y = np.zeros((len(x),))
    std_y = np.zeros((len(x),))
    for ctr in range(len(x)):
         y[ctr] = np.sum(x[ctr:(ctr+N)])/N
         std_y[ctr]=np.sqrt(np.sum(abs(y[ctr]-x[ctr:(ctr+N)])**2)/N)
    return y,std_y

p0ax,p0ay,p0az,p1ax,p1ay,p1az,a0ax,a0ay,a0az=np.loadtxt("acc4624.csv", delimiter=',',skiprows=1,unpack=True)

"""
for i in range(0,len(a0ax)):
    a=np.concatenate((a0ax[i],a0ay[i],a0az[i],np.array([1]))
    k=np.dot(a,opt)
    accxc[i]=k[0]
    accyc[i]=k[1]
    acczc[i]=k[2]
    
"""
#%%
rAx,std_Ax=rolling_average(a0ax,10)
rAy,std_Ay=rolling_average(a0ay,10)
rAz,std_Az=rolling_average(a0ay,10)

rPx0,std_Px0=rolling_average(p0ax,10)
rPy0,std_Py0=rolling_average(p0ay,10)
rPz0,std_Pz0=rolling_average(p0az,10)


accxsp=wavelet_denoise(a0ax,"bior1.5",np.mean(std_Ax))
accysp=wavelet_denoise(a0ay,"bior1.5",np.mean(std_Ay))
acczsp=wavelet_denoise(a0az,"bior1.5",np.mean(std_Az))


apccxsp=wavelet_denoise(p0ax,"bior1.5",np.mean(std_Px0))
apccysp=wavelet_denoise(p0ay,"bior1.5",np.mean(std_Py0))
apcczsp=wavelet_denoise(p0az,"bior1.5",np.mean(std_Pz0))
#k=np.dot(a,opt)

plt.plot(p1ax,'.',label='Gantry 1')
plt.plot(p0ax,'.',label='Gantry 0')
#plt.plot(apccxsp,'.b',label='data processed')
plt.plot(rPx0,'.b',label='data processed')
plt.plot(a0ax,'.',label='A1')

plt.plot(accxsp,'.b',label='data processed')
plt.ylabel("Acceleration (x) (g)")
plt.legend()
plt.show()
plt.plot(p1ay,'.',label='Gantry 1')

plt.plot(p0ay,'.',label='Gantry 0')
plt.plot(rPy0,'.r',label='data processed')
#plt.plot(apccysp,'.b',label='data processed')
plt.plot(a0ay,'.',label='A1')
plt.plot(accysp,'.b',label='data processed')
plt.ylabel("Acceleration (y) (g)")
plt.legend()
plt.show()
plt.plot(p1az,'.',label='Gantry 1')

plt.plot(p0az,'.',label='Gantry 0')
plt.plot(rPz0,'.r',label='data processed')
#plt.plot(apcczsp,'.b',label='data processed')
plt.plot(a0az,'.',label='A1')

plt.plot(acczsp,'.b',label='data processed')
plt.ylabel("Acceleration (z) (g)")
plt.legend()
plt.show()

