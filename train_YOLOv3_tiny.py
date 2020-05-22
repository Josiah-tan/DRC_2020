
#================================================================
#
#   File name   : train_YOLOv3_tiny.py
#   Author      : Josiah Tan
#   Created date: 18/05/2020
#   Description : used to train custom motion planner
#
#================================================================

#================================================================

########## Temporary variables ##############
classes = list(range(4))

#================================================================

"""
TODO:
    change the optimiser for sigmoid output
    rethink the metric - accuracy, perhaps f1 score is more suited for this task
"""


from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import plot_model
from configs import *

base_model = load_model(YOLO_DARKNET_WEIGHTS)

base_model.summary()
exit()
plot_model(base_model, to_file='yolov3_tiny.png', show_shapes=True, show_layer_names=True)
#conv2d_12 (Conv2D) (None, None, None, 2 65535 leaky_re_lu_10[0][0]

x = base_model.get_layer('conv2d_12').output

# Adding a fully-connected layer
x = Dense(1024, activation='relu')(x)

# and a logistic layer -- let's say we have len(classes) = 4 classes
predictions = Dense(len(classes), activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized - glorot apparently)
# i.e. freeze all convolutional yolov3 layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=RMSprop(learning_rate=0.001, rho=0.9), loss='binary_crossentropy', metrics = ["accuracy"]) #categorical_crossentropy? But it is a multiclass problem right?

model.summary()
#plot_model(model, to_file='yolov3_tiny_mod.png', show_shapes=True, show_layer_names=True)
exit()


checkpoint = ModelCheckpoint('best_model.h5', verbose=1, monitor='val_accuracy', save_best_only=True, mode='max') 
#checkpoint = ModelCheckpoint('best_model.h5', verbose=1, monitor='val_accuracy', save_best_only=True, mode='max') 
#checkpoint = ModelCheckpoint('best_model.h5', verbose=1, monitor='acc', save_best_only=True, mode='auto') 


dataAugmentation = ImageDataGenerator(
                horizontal_flip = True,
                vertical_flip = True,
                rescale = 1./255
                )

# train the model on the new data for a few epochs, note that .flow has shuffle = true
history1 = model.fit(dataAugmentation.flow(train_x, train_y, batch_size = 32),
                validation_data = (val_x, val_y), 
                epochs = 10, verbose = 1,
                callbacks = [checkpoint])





"""
# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# and a logistic layer -- let's say we have len(classes) = 5 classes
predictions = Dense(len(classes), activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
#model.compile(optimizer=RMSprop(learning_rate=0.001, rho=0.9), loss='categorical_crossentropy')
model.compile(optimizer=RMSprop(learning_rate=0.001, rho=0.9), loss='categorical_crossentropy', metrics = ["accuracy"])

#model.summary()


checkpoint = ModelCheckpoint('best_model.h5', verbose=1, monitor='val_accuracy', save_best_only=True, mode='max') 
#checkpoint = ModelCheckpoint('best_model.h5', verbose=1, monitor='val_accuracy', save_best_only=True, mode='max') 
#checkpoint = ModelCheckpoint('best_model.h5', verbose=1, monitor='acc', save_best_only=True, mode='auto') 


dataAugmentation = ImageDataGenerator(
                horizontal_flip = True,
                vertical_flip = True,
                rescale = 1./255
                )

# train the model on the new data for a few epochs, note that .flow has shuffle = true
history1 = model.fit(dataAugmentation.flow(train_x, train_y, batch_size = 32),
                validation_data = (val_x, val_y), 
                epochs = 10, verbose = 1,
                callbacks = [checkpoint])

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics = ["accuracy"])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers

# trains for 19 steps (without the floor operation // in len(train_x)//32)
history2 = model.fit(dataAugmentation.flow(train_x, train_y, batch_size = 32),
                validation_data = (val_x, val_y), 
                epochs = 10, verbose = 1,
                callbacks = [checkpoint])
"""