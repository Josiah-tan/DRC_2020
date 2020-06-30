import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
import os
from utils import *

process_image = JTImageProcessing()

path= r"C:\Users\marco\Documents\University\Year2\DRC2020_Anaconda"

#Defines a specific region of interest of the frame and ignores anything outside the mask
def roi(img,vertices):
    mask= np.zeros_like(img)
    cv2.fillPoly(mask,vertices, 255)
    masked= cv2.bitwise_and(img,mask)
    return masked

def RGB2Gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray3 = img[:,:,0] * 0.1 + img[:,:,1] * 0.1 + img[:,:,2] * 0.8
    gray3 = gray3.astype(np.uint8)

    return gray3

def Can(img):
    can_img=cv2.GaussianBlur(img,(5,5),0)
    can_img=cv2.Canny(can_img,threshold1=50,threshold2=200)
    return can_img

#processes the image
def process_img(original_image):
    processed_img=RGB2Gray(original_image)
    processed_img= Can(processed_img)
    #vertices for the mask
    vertices= np.array([[0,480],[0,400],[50,200],[550,200],[600,400],[600,480]])
    processed_img= roi(processed_img,[vertices])
    return processed_img

#function that takes a specific portion of the screen and saves it as a screenshot
def screen_record(x,y,width,height, print_time = False, show_screen = True):
    
    if print_time:
        last_time= time.time()

    #480x600 windowed mode

    #printscreen is the original image
    printscreen= np.array(ImageGrab.grab(bbox=(x,y,width,height)))
    if print_time:
        print('loop took {} sesconds'.format(time.time() - last_time))
        last_time = time.time()

    #frame will be the processed image
    #frame = process_img(printscreen)
    
    printscreen = cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB)

    #change frame size
    #dimensions = (416, 416)
    #resized_image = process_image.image_resize(frame, dimensions)
    if show_screen:
        cv2.imshow('original image',printscreen )
    #cv2.imshow('window', frame)    
    """
    #writes the image 'frame' to the desired folder
    cv2.imwrite(os.path.join(os.path.expanduser('~/Downloads/University/DRC2020_Screen_Recordings'),"test.jpg"), resized_image)
    #jump out of the code if the program is unable to save an image
    if not cv2.imwrite(os.path.join(os.path.expanduser('~/Downloads/University/DRC2020_Screen_Recordings'),"test.jpg"), resized_image):
        raise Exception("Could not write image")
    """
    return printscreen


if __name__ == "__main__":
    while True:
        screen_record(0,40,600,480)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break