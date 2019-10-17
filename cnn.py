import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout
import numpy as np
import pickle
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.regularizers import l1, l2
import time

trainX = np.load('trainX.npy')
trainY = np.load('trainY.npy')

print(trainX.shape)
print(trainY.shape)


convLayers = [3, 4]
convSizes = [64, 128, 256]
denseLayers = [0, 1, 2]
denseSizes = [258, 512]

for convLayer in convLayers:	
	for convSize in convSizes:
		for denseLayer in denseLayers:
			for denseSize in denseSizes:
				name = "({}-conv_{}-units)-({}-dense_{}-units)-{}".format(convLayer, convSize, denseLayer, denseSize, int(time.time()))
				print(name)
				
				tensorboard = TensorBoard(log_dir='logs2/{}'.format(name))
				earlystopping = EarlyStopping(monitor='val_loss', patience=2, mode='auto')

				
				model = Sequential()
				
				model.add(Conv2D(convSize, (3,3), input_shape=trainX.shape[1:]))
				model.add(Activation('relu'))
				model.add(MaxPooling2D(pool_size=(2,2)))

				for l in range(convLayer-1):
					model.add(Conv2D(convSize, (3,3)))
					model.add(Activation('relu'))
					model.add(MaxPooling2D(pool_size=(2,2)))

				model.add(Flatten())
    
				for l in range(denseLayer):
					model.add(Dense(denseSize))
					model.add(Activation('relu'))
					#model.add(Dropout(droprate))

				model.add(Dense(7))
				model.add(Activation('softmax'))


				model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
				model.fit(trainX, trainY, 
					batch_size=32, 
					epochs=10, 
					validation_split=0.1, 
					callbacks=[tensorboard, earlystopping])


# Best models so far
'''
3 conv 128 
1 dense 512
6 epochs 
loss = 1.239

3 conv 128
0 dense
6 epochs
loss = 1.219




'''


