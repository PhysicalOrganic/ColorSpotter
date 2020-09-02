# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:23:06 2020

@author: Sarah Moor
"""


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb
import matplotlib
import matplotlib.pyplot as plt


from tkinter import filedialog
from tkinter import *


def main ():  
   # open file finder to prompt user to choose image
   root = Tk()
   root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
   root.destroy()
   
   TLC = cv2.imread(root.filename)
   TLC = cv2.cvtColor(TLC, cv2.COLOR_BGR2RGB)
   
   # show image chosen
   plt.imshow(TLC)
   plt.show()
   
   # split the 3 values in RGB to create list
   r, g, b = cv2.split(TLC)
   
   # create customizable plot font options
   SMALL_SIZE = 9
   MEDIUM_SIZE = 14
   BIGGER_SIZE = 14
   
   plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
   plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
   plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
   plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
   plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
   plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
   plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
     

   # Commenting out unfiltered plots that include color of the TLC plate RGB
   '''  
   fig = plt.figure()
   axis = fig.add_subplot(1, 1, 1, projection="3d")
   pixel_colors = TLC.reshape((np.shape(TLC)[0] * np.shape(TLC)[1], 3))
   norm = colors.Normalize(vmin=-1.0, vmax=1.0)
   norm.autoscale(pixel_colors)
   pixel_colors = norm(pixel_colors).tolist()
        
   axis.scatter(
       r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker="."
       )
   axis.set_xlabel("Red")
   axis.set_ylabel("Green")
   axis.set_zlabel("Blue")
   # plt.show()
   ''' 
   
   # split HSV color scheme into list
   hsv_TLC = cv2.cvtColor(TLC, cv2.COLOR_RGB2HSV)
    
   h, s, v = cv2.split(hsv_TLC)
   
   
   # Commenting out unfiltered plots that include color of the TLC plate HSV     
   '''
   fig = plt.figure()
   axis = fig.add_subplot(1, 1, 1, projection="3d")
        
   axis.scatter(
       h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker="."
       )
   axis.set_xlabel("Hue")
   axis.set_ylabel("Saturation")
   axis.set_zlabel("Value")
   # plt.show()
   '''
   
   # Color range of TLC plate to bitwise filter out
   light = (0,60,120)
   dark = (175,200,200)

    
   mask = cv2.inRange(hsv_TLC, light, dark)
    
   # Bitwise-AND mask and original image  
   result = cv2.bitwise_and(TLC, TLC, mask=mask)   

   # replotting after filter
   TLC = result
    
   r, g, b = cv2.split(TLC)
    
   # adjust figure size
   fig = plt.figure(figsize=(9,9.5))
   
   # plot image
   axis = fig.add_subplot(1, 1, 1, projection="3d")
   pixel_colors = TLC.reshape((np.shape(TLC)[0] * np.shape(TLC)[1], 3))
   norm = colors.Normalize(vmin=-1.0, vmax=1.0)
   norm.autoscale(pixel_colors)
   pixel_colors = norm(pixel_colors).tolist()
    
   axis.scatter(
       r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker="."
   )
   
   # Set axis labels
   axis.set_xlabel("Red", labelpad=10)
   axis.set_ylabel("Green", labelpad=10)
   axis.set_zlabel("Blue", labelpad=10)
   plt.show()
    
   hsv_TLC = cv2.cvtColor(TLC, cv2.COLOR_RGB2HSV)
    
   h, s, v = cv2.split(hsv_TLC)
    
   fig = plt.figure(figsize=(9, 9.5))
   axis = fig.add_subplot(1, 1, 1, projection="3d")
    
   axis.scatter(
       h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker="."
   )
   
   # Set axis llabels
   axis.set_xlabel("Hue", labelpad=10)
   axis.set_ylabel("Saturation", labelpad=10)
   axis.set_zlabel("Value", labelpad=10)
   plt.show()   
   
    
main()