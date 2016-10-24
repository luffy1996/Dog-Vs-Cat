'''This is assignment 2 , from Advanced Machine Learning Course by Professor Amit Sethi'''
from __future__ import print_function
import numpy as np
import time
np.random.seed(2001)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten , Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import scipy.io as sio

def make_validation(train_data,train_target):
	test_data=np.zeros(shape=(4000,3,64,64),dtype=np.float32)
	test_target=np.zeros(shape=(4000,1),dtype=np.float32)
	print(test_data.dtype)
	print(test_target.dtype)
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
	print(s)
	print('Size of testing data is '+ str(test_data.shape))
	print('Size of testing data target '+ str(test_target.shape))
	return(test_data,test_target)	



def load_data(batch_size=500,nb_classes=2,nb_epoch=100):
	traindata = sio.loadmat('traindata.mat')
	trainX = traindata['trainX']
	trainX = np.reshape(trainX,(trainX.shape[0],3,64,64))
	print('Size of training data is '+ str(trainX.shape))


	trainY = traindata['trainY']
	print('Size of training data target is '+ str(trainY.shape))

	testdata = sio.loadmat('testdata.mat')
	testX = testdata['testX']
	testX = np.reshape(testX,(testX.shape[0],3,64,64))
	(testX,testY)=make_validation(trainX,trainY)





if __name__ == '__main__':
	load_data()
	print ('Done')
	time.sleep(3)

