
#================================================================
#
#   File name   : utils.py
#   Author      : Josiah Tan
#   Created date: 22/05/2020
#   Description : Random utilities for deep learnin stuff
#
#================================================================

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


class JTImageProcessing:
    def print_image(self, image, cmap = "Greys"):
        """
        parameters -- imageL takes in a list of np.arrays or a dictionary containing key with array_name and the value of array itself
        returns none
        
        loads the images onto a figure
        examples:
            print_image([resize_image, image]) # plot without labels
            print_image({"resized_image" : resized_image, "image" : image}) #plot with labels
        """
        
        """
        NOTE: cmap is ignored for RGB data
        """
        
        if type(image)==list:
            #print (image[0].shape)
            fig = plt.figure(figsize = (8,8))
            for i in range(0,len(image)):
                fig.add_subplot(2,4,1 +i)
                plt.imshow(image[i], cmap = cmap)
            plt.show()
        elif type(image) == dict:
            fig = plt.figure(figsize = (8,8))
            i = 1
            for key, val in image.items():
                fig.add_subplot(2,4,i)
                i+=1
                plt.imshow(val, cmap = cmap)
                plt.xlabel(key)
            plt.show()
            
    def file2array(self, file_name):
        """converts a file to an array"""
        return np.array(plt.imread(file_name))
        return np.array(raw_image)
        
    def image_resize(self, raw_image, dimensions, method = "nearest neighbour"):
        """
        parameters -- raw_image - a numpy array of size y, x, n or y,x
                   -- dimensions - a tuple containing the resized dimensions
                                 - example: (416, 416, 3) or (150, 150)
                   -- method - method of resizing
        """
        resized_image = np.array(Image.fromarray(raw_image).resize((dimensions[1],dimensions[0]))) # resize image - nearest neighbour technique
        return resized_image


class JTTestCases(JTImageProcessing):
    def __init__(self):
        pass
        
    def test_image_resize_from_file(self, file_name, dimensions):
        """Tests the image_resize method by printing out the resized image"""
        raw_image = self.file2array(file_name)
        
        
        resized_image = self.image_resize(raw_image, dimensions)
        
        self.print_image({"resized_image" : resized_image, "raw_image" : raw_image})
    
    def test_image_resize(self, raw_image, dimensions):
        resized_image = testing.image_resize(raw_image, dimensions)
        testing.print_image({"resized_image" : resized_image, "raw_image" : raw_image})
if __name__ == "__main__":
    testing = JTTestCases()
    
    file_name = "aphid_train.jpg"
    dimensions = (150,150)
    
    testing.test_image_resize_from_file(file_name, dimensions)
    
    
    raw_image = cv2.imread(file_name, 0)
    
    testing.test_image_resize(raw_image, dimensions)
    
    