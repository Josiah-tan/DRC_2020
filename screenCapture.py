import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
import os

path= r"C:\Users\marco\Documents\University\Year2\DRC2020_Anaconda"

def roi(img,vertices):
    mask= np.zeros_like(img)
    cv2.fillPoly(mask,vertices, 255)
    masked= cv2.bitwise_and(img,mask)
    return masked

def process_img(original_image):
    processed_img=cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img=cv2.GaussianBlur(processed_img,(5,q),0)
    processed_img= cv2.Canny(processed_img,threshold1=50,threshold2=150)
    vertices= np.array([[0,480],[0,400],[50,200],[550,200],[600,400],[600,480]])
    processed_img= roi(processed_img,[vertices])
    return processed_img

def screen_record():
    last_time= time.time()
    i=1

    dimensions= (480,480)
    while(True):
        #480x600 windowed mode
        printscreen= np.array(ImageGrab.grab(bbox=(0,40,600,480)))
        print('loop took {} seconds'.format(time.time() - last_time))
        last_time = time.time()
        frame = process_img(printscreen)
        cv2.imshow('window', printscreen)
        cv2.imshow('window', frame)
        cv2.imwrite(os.path.join(os.path.expanduser('~/Downloads/University/DRC2020_Screen_Recordings'),str(i)+"test.jpg"), frame)
        if not cv2.imwrite(os.path.join(os.path.expanduser('~/Downloads/University/DRC2020_Screen_Recordings'),str(i)+"test.jpg"), frame):
            raise Exception("Could not write image")

        i=i+1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
'''
DRC2020_Screen_Recordings
test_image=cv2.imread('zukoLightning.jpg')
test_image=np.array(test_image)
print(test_image)
cv2.imwrite(os.path.join(os.path.expanduser('~'),'Desktop',"test.bmp"), test_image)
if not cv2.imwrite(os.path.join(os.path.expanduser('~'),'Desktop',"test.bmp"), test_image):
    raise Exception("Could not write image")
'''
screen_record()
'''
print("Current directory is %s" % path)
print("abspath is %s" %os.path.abspath(path))
print("dirname is %s" %os.path.dirname(path))
print("basename is %s" %os.path.basename(path))
'''



