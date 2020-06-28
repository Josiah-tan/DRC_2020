import interface_edited
import cv2

def controller(image, speed, angle):
    cv2.imshow("car view",image)
    cv2.waitKey(1)

    print("Speed {}, Angle {}".format(speed, angle))

    controlCommand={}

    controlCommand["speed"]= speed
    controlCommand["steer"]= angle
    return controlCommand

interface_edited.onMessage(controller)

interface_edited.start()
