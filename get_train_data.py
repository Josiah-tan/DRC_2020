#================================================================
#
#   File name   : get_train_data.py
#   Author      : Josiah Tan
#   Created date: 23/06/2020
#   Description : Records keyboard input and the screen
#
#================================================================

#================================================================

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import joblib
import pandas as pd
import cv2
from PIL import Image
import os
import h5py

#importing Josiah's image processing module (I should really change the name utils...)
import utils

   

#import Marco's/ Sharon's modules
import screenCapture

# configuration for screen capture
width = 640
height = 360
#x_screen_corner = 1280 / 2 - width / 2
#y_screen_corner = 720 / 2 - height / 2
x_screen_corner = 0
y_screen_corner = 50
print_time = False
show_screen = False

# configuration for black and white
gray_dims = (100, 100)
gray_output_filename = "gray_datasets.h5"

#configuration for RGB
RGB_dims = (416, 416)
RGB_output_filename = "RGB_datasets.h5"

#configuration for data
classes = ["right", "left", "up", "down"]
exclude_forward = False # exclude the data np.array([0,0,1,0])

processimg_obj = utils.JTImageProcessing()
processimg_obj.append_data(classes, "classes")

# Echo server section
import socket

HOST = ''                 # Symbolic name meaning all available interfaces
PORT = 5000              # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
conn, addr = s.accept()
print('Connected by', addr)
# end echo server section

while True:
    data = conn.recv(1024)
    if not data: break
    raw_data = data.decode()
    print(raw_data) # note that raw_data sometimes returns 
    
    y = np.array([b == "1" for b in raw_data.split('\n')[0].split()]) # store data here "0 1 0 1"
    print(y.shape)
    assert y.shape[0] == 4
    # do whatever you need to do with the data

    
    screen_capture = screenCapture.screen_record(x_screen_corner, y_screen_corner, width, height, print_time, show_screen)
    
    gray_img = screenCapture.RGB2Gray(screen_capture)
    
    resized_gray = processimg_obj.image_resize(gray_img, gray_dims)
    
    resized_rgb = processimg_obj.image_resize(screen_capture, RGB_dims)
    
    if cv2.waitKey(25) & 0xFF == ord('q'): # for breaking out of the loop when q is pressed
        cv2.destroyAllWindows()
        break
    """
    if y[0] == 0 and y[1] == 0 and y[2] == 1 and y[3] == 0 and exclude_forward: #skip over y = np.array([0,0,1,0])
        continue
    """
    processimg_obj.append_data(resized_gray, "gray_x") 
    processimg_obj.append_data(resized_rgb, "rgb_x")
    processimg_obj.append_data(y, "raw_y")
    #plt.imshow(resized_rgb, "gray")
    #plt.show()
    
processimg_obj.create_whole_dataset(key_x = "gray_x", key_y = "raw_y", key_classes = "classes", username = "Josiah", dataset_file_name = gray_output_filename, mode = 'a')
processimg_obj.create_whole_dataset(key_x = "rgb_x", key_y = "raw_y", key_classes = "classes", username = "Josiah", dataset_file_name = RGB_output_filename, mode = 'a')

conn.close()
# optionally put a loop here so that you start 
# listening again after the connection closes

