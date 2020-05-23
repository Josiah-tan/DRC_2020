
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
import os
import h5py


class Debugger:
    """ used when I can't be bothered to comment stuff out when printing"""
    def __init__(self):
        self.__display = False
    def disp(self, something, manual_display_mode = True):
        if self.__display:
            if manual_display_mode:
                print(something)
    def change_disp_mode(self, display_mode):
        #set the default display mode
        self.__display = display_mode


class JThdf(Debugger):
    def __init__(self):
        self.__data = {}
        super().__init__()
    def __call__(self, file_name):
        self.__file_name = file_name
        
    
    def create_new_datasets(self, data, name = "", username = "", dataset_file_name = "", mode = 'a'): 
        """
        WARNING: this function can override data, so be careful, only use 'w' when u know what u r doing
        parameters -- data is a list of arrays
                   -- name is a string containing the name of the dataset
        returns -- none
        
        writes data to .h files
        """
        self.disp("CREATE NEW DATASETS")
        
        if dataset_file_name == "":
            dataset_file_name = self.__file_name
        
        with h5py.File(dataset_file_name, mode) as hdf: #note that 'w' actually erases the contents of the file
            user_group = hdf.require_group(username)
            type_group = user_group.require_group(name)
            type_group_items = list(type_group.items())
            type_group_keys = list(type_group.keys())
            self.disp(f"type_group_items {type_group_items}")
            self.disp(f"type_group_keys {type_group_keys}")
            num_items = str(len(type_group_keys))
            type_group.create_dataset(num_items, data = data)
            
            """
            #subgroup = hdf.require_group(f"{username}_subgroup_{name}")
            #hdf.create_dataset("random", data = data)
            ls = dict(hdf.keys())
            print(ls)
            
            hdf.create_dataset('set_x', data = np.array(x_lis)) #x_lis should be a list containing np.arrays
            hdf.create_dataset('set_y', data = np.array(y_lis))
            hdf.create_dataset('list_classes', data = np.array([x.encode('utf-8') for x in list_classes]))
            ls = list(hdf.keys()) 
            print(f"List of datasets in this file: \n {ls}")
            
            x_orig = np.array(hdf.get('set_x'))
            y_orig = np.array(hdf.get('set_y')).reshape((1,-1)) # changes (8,) to(1,8)
            classes = np.array(hdf.get('list_classes'))
            """
    
    def read_datasets(self, name = "", username = "", index = "all", dataset_file_name = "", mode = "r"):
        """
        NOTE: use np.concatenate(complete_dataset, axis = 0) to combine all the data in the keys
        parameters -- name: this is the name of the dataset - could be "raw_x"
                   -- username: this is the username of the dataset - has to be your name
                   -- index: the default of index is all, however the dataset can be accessed via an index  such as 0, 1 or -1 for the last index
                   -- dataset_file_name: defaults to the file name which results from using the __call__ method if left at "", otherwise it reads a file that you specify
                   -- mode: this is the mode at which you are reading the file, keep this at "r" unless u know what you are doing
        returns -- returns the dataset as specified by the parameters    
        
        setup -- dataset_file_name.h
                    -username
                        -- name
                            -- 0
                            -- 1
                            -- 2
                
        """
        
        self.disp("Reading Datasets")
        if dataset_file_name == "":
            dataset_file_name = self.__file_name # sets the name to default file name specified by __call__ method
        with h5py.File(dataset_file_name, mode) as hdf:
            user_group = hdf.get(username) # goes into the username directory such as "raw_y"
            type_group = user_group.get(name) # goes into the name directory such as "raw_x"
            keys = list(type_group.keys()) # checks out the names of the datasets in the name directory
            if index == "all":
                return_data = [] # list to store all the datasets within the name directory
                for key in keys:
                    dataset = type_group.get(key)
                    self.disp(f"{key}: {dataset.shape}")
                    return_data.append(dataset)
                complete_dataset = np.array(return_data)
                self.disp(f"complete dataset.shape: {complete_dataset.shape}")
                return complete_dataset
            else:
                key = keys[index] # accessing a name via the index specified as a parameter
                complete_dataset = np.array(np.array(type_group.get(key)))
                self.disp(f"{key}: {complete_dataset.shape}") # displaying the shape of the returned dataset
                return complete_dataset
    
    def create_whole_dataset(self, x = None, y = None, classes = None, key_x = "raw_x", key_y = "raw_y", key_classes = "classes", username = "", dataset_file_name = "", mode = 'a'):
        
        """Using appended/ default values"""
        if x == None:
            x = self.__data[key_x]
        if y == None:    
            y = self.__data[key_y]
        if classes == None:
            classes = self.__data[key_classes] 
        if dataset_file_name == "":
            dataset_file_name = self.__file_name
            
            
        
        with h5py.File(self.__file_name, mode) as hdf:
            self.disp(list(hdf.keys()))
            if key_classes not in list(hdf.keys()):
                hdf.create_dataset(key_classes, data = np.array([arrow_key.encode('utf-8') for arrow_key in classes])) # creating dataset for classes
        self.create_new_datasets(x, name = key_x, username = username, dataset_file_name = dataset_file_name, mode = 'a') # create dataset for x
        self.create_new_datasets(y, name = key_y, username = username, dataset_file_name = dataset_file_name, mode = 'a') # create dataset for y
    
    def read_whole_dataset(self, key_x = "raw_x", key_y = "raw_y", key_classes = "classes", username = "", dataset_file_name = "", mode = 'r'):
        if dataset_file_name == "":
            dataset_file_name = self.__file_name
        with h5py.File(dataset_file_name, mode) as hdf:
            classes = np.array(hdf.get(key_classes))
            classes = [x.decode('utf-8') for x in classes]
        x = self.read_datasets(name = key_x, username = username, index = "all", dataset_file_name = dataset_file_name, mode = "r")    
        y = self.read_datasets(name = key_y, username = username, index = "all", dataset_file_name = dataset_file_name, mode = "r")
        return x, y, classes
    
    def append_data(self, data, name = ""):
        self.disp("Appending to Datasets")
        if name in self.__data:
            self.__data[name].append(data)
        else:
            if type(data) == list:
                self.__data[name] = data # had "classes" list in mind when creating this
            else:
                self.__data[name] = [data]            
class JTImageProcessing(JThdf):
    def __init__(self):
        super().__init__()
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
        super().__init__()
        
        program_number = -1
        
        programs = ("test JThdf __call__", "test_image_resize_from_file", "test_image_resize", "test_create_new_datasets", "test_append_data", "test_read_datasets", "test_whole_datasets", "test_user_interface")
        
        self.image_file_name = "aphid_train.jpg"
        self.dimensions = (150,150)
    
        self.output_file_name = r"C:\Users\josia\Desktop\Josiah_Folder\UNI\Semester_1\PEP1\robotics_club\random_datasets.h5"
        
        self.raw_image = cv2.imread(self.image_file_name, 0)
        
        program = programs[program_number]
        
        self.disp(self.exc_program_chooser(program))
    
    def exc_program_chooser(self, program):
        return {
            'test JThdf __call__': self.test_hdf_call,
            'test_image_resize_from_file': self.test_image_resize_from_file,
            'test_image_resize': self.test_image_resize,
            'test_create_new_datasets': self.test_create_new_datasets,
            'test_append_data': self.test_append_data,
            'test_read_datasets' : self.test_read_datasets,
            'test_whole_datasets' : self.test_whole_datasets,
            'test_user_interface' : self.test_user_interface
        }.get(program, lambda: None)()
    
    def test_image_resize_from_file(self):
        """Tests the image_resize method by printing out the resized image"""
        raw_image = self.file2array(self.image_file_name)
        
        
        resized_image = self.image_resize(raw_image, self.dimensions)
        
        self.print_image({"resized_image" : resized_image, "raw_image" : raw_image})
        return raw_image, resized_image
    def test_image_resize(self):
        resized_image = testing.image_resize(self.raw_image, self.dimensions)
        self.print_image({"resized_image" : resized_image, "raw_image" : self.raw_image})
        
    def test_hdf_call(self):
        file_name = "hoho.jpg"
        #print(dir(self))
        #print(self._JThdf__file_name)
        self(file_name)
        print(self._JThdf__file_name)
        
        file_name = "datasets.h"
        self(file_name)
        print(self._JThdf__file_name)
        
    def test_create_new_datasets(self):
        self(self.output_file_name)
        x = 0
        
        CURRENT_PATH = os.getcwd()
        
        os.chdir("..")
        
        sample_photos_dir = os.getcwd() + "\Aphid_pics"
        
        #os.chdir(CURRENT_PATH)
        print(os.getcwd())
        x_raw = []
        for image_files in os.listdir(sample_photos_dir):
            raw_image = self.file2array(os.path.join(sample_photos_dir, image_files))        
            x_raw.append(self.image_resize(raw_image, self.dimensions))
        self.create_new_datasets(x_raw, name = "raw_x", username = "Josiah")
        
        """
            if x == 7:
                break
            x +=1
        self.print_image(resized_image)
        """
    def test_append_data(self):
        self.change_disp_mode(0)
        (raw_image, resized_image) = self.test_image_resize_from_file()
        self(self.output_file_name)
        
        
        self.disp(resized_image.shape)
        for _ in range(10):
            self.append_data(resized_image, name = "raw_x")
        
        #self.append_data(resized_image, name = "raw_x")
        self.disp(len(self._JThdf__data["raw_x"]))
        self.disp(resized_image.shape)

    def test_read_datasets(self):
        self(self.output_file_name)
        self.change_disp_mode(1)
        #a = self.read_datasets(name = "x_raw", username = "Josiah", index = 2, dataset_file_name = "")
        a = self.read_datasets(name = "x_raw", username = "Josiah", index = "all", dataset_file_name = "")
        self.disp(f"a.shape: {a.shape}")
        
    def test_whole_datasets(self):
        self(self.output_file_name)
        self.change_disp_mode(1)
        """
        x = []
        sample_photos_dir = 
        for image_files in os.listdir(sample_photos_dir):
            raw_image = self.file2array(os.path.join(sample_photos_dir, image_files))        
            x.append(self.image_resize(raw_image, self.dimensions))
        """
        x = [np.random.randn(150,150,3) for I in range(293)]
        y = (np.random.randn(293, 4) < 0.3).astype("int")
        classes = ["right", "left", "up", "down"]
        self.create_whole_dataset(x, y, classes, key_x = "raw_x", key_y = "raw_y", key_classes = "classes", username = "Josiah")
        x,y, classes = self.read_whole_dataset(key_x = "raw_x", key_y = "raw_y", key_classes = "classes", username = "Josiah")
        self.disp(f"classes: {classes}")
        self.disp(f"x.shape: {x.shape}")
        self.disp(f"y.shape: {y.shape}")
    def test_user_interface(self):
        self(self.output_file_name)
        self.change_disp_mode(1)
        x = self.file2array("aphid_train.jpg")
        y = (np.random.randn(1,4) <= 1/4).astype('int')
        classes = ["right", "left", "up", "down"]
        self.disp(f"x.shape {x.shape}")
        self.disp(f"y {y}")
        for _ in range(293):
            self.append_data(x, "raw_x")
            self.append_data(y, "raw_y")
        
        self.append_data(classes, "classes")
        self.disp(self._JThdf__data)
        self.create_whole_dataset(key_x = "raw_x", key_y = "raw_y", key_classes = "classes", username = "Bob", dataset_file_name = "", mode = 'a')
        
if __name__ == "__main__": 
    testing = JTTestCases()
    
    
    
    
    
    
    
    
    
    
    
    