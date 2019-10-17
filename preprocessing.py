import numpy as np 
import matplotlib.pyplot as plt 
import os
import cv2 # For image operations
import random
import pandas as pd

# --------- METHODS ----------
def getImageArray(pixels):
	final = []
	array = pixels.split()
	for i in range(len(array)):
		array[i] = int(array[i])
	#print(array)
	#print(array[:48])
	for i in range(48):
		final.append(array[48*i:48*i+48])
	return final


# --------- SCRIPT ----------

path = '/Users/axeloh/Koding/machine_learning/datasets/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013/fer2013.csv'
df = pd.read_csv(path, sep=',')
# Emotion: (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
# Usage: (Training, PublicTest, PrivateTest)


trainingData = []
testData = []

for index, row in df.iterrows():
	label = row['emotion']
	pixels = row['pixels']
	usage = row['Usage']
	imgArray = getImageArray(pixels)
	#plt.imshow(imgArray, cmap="gray")
	#plt.show()
	data = [imgArray, label]
	if usage == 'Training':
		trainingData.append(data)
	elif usage == 'PublicTest':
		testData.append(data)



random.shuffle(trainingData)

trainX = []
trainY = []

testX = []
testY = []

for features, label in trainingData:
	trainX.append(features)
	trainY.append(label)

for features, label in testData:
	testX.append(features)
	testY.append(label)


trainX = np.array(trainX).reshape(-1, 48, 48, 1)
testX = np.array(testX).reshape(-1, 48, 48, 1)

trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)

trainX = trainX/255.0
testX = testX/255.0


print(trainX.shape)
print(trainY.shape)
print(testX.shape)


print("Saving data..")

np.save('./trainX.npy', trainX)
np.save('./trainY.npy', trainY)
np.save('./testX.npy', testX)
