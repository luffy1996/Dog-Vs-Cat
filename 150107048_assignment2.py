'''This is assignment 2 , from Advanced Machine Learning Course by Professor Amit Sethi'''
from __future__ import print_function
import numpy as np
import scipy.io as sio
import time
np.random.seed(2001)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.regularizers import l2,l1,l1l2,activity_l2
from keras.models import load_model
from keras.models import model_from_json


def load_data(batch_size=600,nb_classes=2,nb_epoch=20,nb_filters=(50,100)):
	print('... loading data')
	traindata = sio.loadmat('traindata.mat')
	trainX = traindata['trainX']
	trainX = np.reshape(trainX,(trainX.shape[0],3,64,64))
	print('Size of training data is '+ str(trainX.shape))


	trainY = traindata['trainY']
	print('Size of training data target is '+ str(trainY.shape))

	trainX=trainX.astype('float32')
	trainY=trainY.astype('int32')
	train_X= trainX
	trainX /= 255
	print('X train shape:', train_X.shape)

	print ('data loaded')
	###########################################################
	pool_size=(2,2)
	kernel_size0=(5,5)
	kernel_size1=(7,7)
	# convert class vectors to binary class matrices					
	trainY = np_utils.to_categorical(trainY,nb_classes)

	#######################################
	###  Training Model ###################
	#######################################
	model = Sequential()
	model.add(Convolution2D(nb_filters[0],kernel_size0[0], kernel_size0[1],border_mode='valid',input_shape=(3,64,64),name='layer0'))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=pool_size,name='layer1'))
	model.add(Convolution2D(nb_filters[1] ,kernel_size1[0], kernel_size1[1],name='layer2'))
	model.add(LeakyReLU(alpha=.0015))
	model.add(MaxPooling2D(pool_size=pool_size,name='layer3'))
	model.add(Flatten())
	model.add(Dense(100,W_regularizer=l2(0.001),activity_regularizer=activity_l2(0.001),name='layer4'))
	model.add(BatchNormalization())
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes,name='layer5'))
	model.add(Activation('softmax'))
	model.load_weights("model.h5")
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.fit(train_X, trainY, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_split=.40)
	print('fit ended')
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
		# serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved model to disk")


def predict():
	testing_data = sio.loadmat('testdata.mat')
	testing_X = testing_data['testX']
	testing_X = np.reshape(testing_X,(testing_X.shape[0],3,64,64))

	print ('testing data loaded')	
	###############################################################

	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model

	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")
	target=loaded_model.predict(testing_X)
	print(target[11000:11030,])
	#text_file = open("Output.txt", "w")
	#text_file.write(target)
	#text_file.close()
	
	################################################################

	#saving to text file
	text_file = open("Output_150107048.txt", "w")
	for i in range(target.shape[0]):
		text_file.write(str(target[i]))
		text_file.write("\n")
	text_file.close()

	val = 0
	text_file = open("Output_150107048_dc.txt", "w")
	for i in range(target.shape[0]):
		if (target [i,0]>=0.5 ):
			val = 0
		else :
			val = 1

		text_file.write(str(val))
		text_file.write("\n")
	text_file.close()

if __name__ == '__main__':
	'''If you want to train data yourself uncomment load_data()'''
	#load_data()
	predict()
	print ('Done')
	time.sleep(3)

