import pyautogui
import numpy as np

def press_keys(array): #[w,a,s,d]
    if array[0]:
        pyautogui.keyDown("d")
    else:
        pyautogui.keyUp("d")
        
    if array[1]:
        pyautogui.keyDown("a")
    else:
        pyautogui.keyUp("a")
        
    if array[2]:
        pyautogui.keyDown("w")
    else:
        pyautogui.keyUp("w")
        
    if array[3]:
        pyautogui.keyDown("s")
    else:
        pyautogui.keyUp("s")

if __name__ == "__main__":
    press_keys(np.array([True,True,False,False]))
    
    
