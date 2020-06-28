# now the websocket
import websockets
import asyncio
import cv2
import numpy as np
import json


onMessageFunction=None
def onMessage(f):
    global onMessageFunction
    onMessageFunction=f

async def middleman(websocket, path):


    while (1):
        speed = 0
        rotation = 0
        speed_increment = 200
        rotation_increment = 30

        dir = input()

        if dir == "w":
            speed += speed_increment
        elif dir == "s":
            speed += speed_increment
        elif dir == "a":
            speed += speed_increment/2
            rotation -= rotation_increment
        elif dir == "d":
            speed += speed_increment/2
            rotation += rotation_increment


        data = await websocket.recv()
        img= np.frombuffer(data,dtype=np.uint8)
        array= np.reshape(img,(480,640,4))[:,:,0:3]
        (b,g,r)= cv2.split(array)
        array= cv2.merge([r,g,b])
        array= np.flip(array,0)
        controlTuple= onMessageFunction(array, speed, rotation)
        await websocket.send(json.dumps(controlTuple))

def start():
    start_server = websockets.serve(middleman, "localhost", 8081, max_size=2**30)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
