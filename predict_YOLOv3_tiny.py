    
    
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from configs import *    


num_px = 416

file_name = "aphid_train.jpg"


image = np.array(plt.imread(file_name))
    
base_model = load_model(YOLO_DARKNET_WEIGHTS)
trained_model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv2d_12').output)
resized_image = np.array(Image.fromarray(image).resize((num_px,num_px)))
"""
#testing if images have been resized correctly
fig = plt.figure(figsize = (8,8))
fig.add_subplot(1,2,1)
plt.imshow(image)
fig.add_subplot(1,2,2)
plt.imshow(resized_image)
plt.show()
"""
#image_flatten = resized_image.reshape(1, -1).T

#image_standardised = image_flatten/255.
image_standardised = resized_image/255.

#print("checking the resized shape")
#print ("image_standardised's shape: " + str(image_standardised.shape))

#print(f"layer dimensions: {layers_dims}")

#probabilities, _ = insect_dnn.L_model_forward_multiclass(image_standardised, parameters)


image_expanded = np.expand_dims(image_standardised, axis=0) # convert (num_px, num_px, 3) to (1, num_px, num_px, 3)
print(f"image_expanded.shape {image_expanded.shape}")
probabilities = trained_model.predict(image_expanded, verbose = 1) # returns numpy array

print(probabilities.shape)