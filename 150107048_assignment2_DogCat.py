'''This is assignment 2 , from Advanced Machine Learning Course by Professor Amit Sethi'''
from __future__ import print_function
import numpy as np
import time
np.random.seed(2001)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD

import scipy.io as sio

def make_test_data(train_data,train_target):
	test_data=np.zeros(shape=(4000,3,64,64),dtype=np.float32)
	test_target=np.zeros(shape=(4000,1),dtype=np.float32)
	#print(test_data.dtype)
	#print(test_target.dtype)
	j=0
	cat=dog=0
	for i in range(len(train_target)):
		if j==3999:
			#print(("value of i is %d"+i))
			break

		if (train_target[i,]==0 and cat>=2000):
			continue
		if (train_target[i,]==1 and dog>=2000):
			continue
		else:
			test_data[j,]=train_data[i,]
			test_target[j,]=train_target[i,]
			#test_set[j,]=i 
			j=j+1
			if (train_target[i,]==0):
					cat=cat+1
			else :
				dog=dog+1
		s=0
	for i in range(len(test_target)):
		if test_target[i,]==0 :
			s =s +1
	#print(s)
	print('Size of testing data is '+ str(test_data.shape))
	print('Size of testing data target '+ str(test_target.shape))
	return(test_data,test_target)	
	''' I have made this funtion to make new testing set '''	


def load_data(batch_size=2500,nb_classes=2,nb_epoch=100,nb_filters=(10,20,40)):
	print('... loading data')
	traindata = sio.loadmat('traindata.mat')
	trainX = traindata['trainX']
	trainX = np.reshape(trainX,(trainX.shape[0],3,64,64))
	print('Size of training data is '+ str(trainX.shape))


	trainY = traindata['trainY']
	print('Size of training data target is '+ str(trainY.shape))

	testdata = sio.loadmat('testdata.mat')
	testX = testdata['testX']
	testX = np.reshape(testX,(testX.shape[0],3,64,64))
	(testX,testY)=make_test_data(trainX,trainY)

	trainX=trainX.astype('float32')
	trainY=trainY.astype('int32')
	testX=testX.astype('float32')
	testY=testY.astype('int32')

	train_X= trainX.transpose(0,2,3,1)
	test_X =testX.transpose(0,2,3,1)
	trainX /= 255
	testX /= 255
	train_X /= 255
	test_X /= 255
	print('X train shape:', train_X.shape)
	print('X test shape:', test_X.shape)

	print ('data loaded')
	###########################################################
	pool_size=(2,2)
	kernel_size0=(3,3)
	kernel_size1=(3,3)
	#kernel_size2=(4,4)

	# convert class vectors to binary class matrices					
	trainY = np_utils.to_categorical(trainY,nb_classes)
	testY = np_utils.to_categorical(testY,nb_classes)
	print(trainX.shape[0])

	model = Sequential()
	model.add(Convolution2D(nb_filters[0],kernel_size0[0], kernel_size0[1],border_mode='valid',
		input_shape=(64,64,3)))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=pool_size))
	#model.add(Activation('relu'))

	model.add(Convolution2D(nb_filters[1] ,kernel_size1[0], kernel_size1[1],border_mode='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))
	#model.add(Activation('relu'))
	
	#model.add(Convolution2D(nb_filters[2],kernel_size2[0], kernel_size2[1],border_mode='valid'))
	#model.add(Activation('relu'))
	#model.add(MaxPooling2D(pool_size=pool_size))
	#model.add(Dropout(0.25))


	model.add(Flatten())
	model.add(Dense(500))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(50))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	
	model.add(Dense(nb_classes))
	model.add(Activation('sigmoid'))

	model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

	model.fit(train_X, trainY, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(test_X, testY))
	#verbose explain
	score = model.evaluate(testX, testY, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])









if __name__ == '__main__':
	load_data()
	print ('Done')
	time.sleep(3)

