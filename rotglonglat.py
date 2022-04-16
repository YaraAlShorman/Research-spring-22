#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 19:04:52 2022

@author: rwatkins
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

data= np.load("C:/Users/yaras/Documents/Research/Feldman/outerrim/CF3-OuterRim-CF3grouplike-cz-LG/CF3-OuterRim-CF3grouplike-cz-LG-box-263.npy")

glon= data[: ,10] * np.pi / 180
glat= data[: ,11] * np.pi / 180

#make glon and glat into array of positions on the unit sphere
pos= np.array([np.cos(glon) * np.cos(glat),  # alpha
               np.sin(glon) * np.cos(glat),  # beta
               np.sin(glat)])                # gamma

pos= np.transpose(pos)

alpha= 2 * np.pi * np.random.random()  #generate random rotation angles
gamma= 2 * np.pi * np.random.random()
beta= np.pi * np.random.random()

r = R.from_euler('xzx', [alpha, beta, gamma]) #create the rotation

posprime=r.apply(pos)  #apply the rotation to the vectors

glonprime= np.arctan2(posprime[: ,1],posprime[: ,0]) * 180 / np.pi  #convert back to glon and glat
glonprime[glonprime < 0]= 360 + glonprime[glonprime < 0]

glatprime= np.arcsin(posprime[: ,2]) * 180/np.pi
                    
plt.plot(glonprime,glatprime,'.')
plt.plot(glonprime[:2], glatprime[:2])