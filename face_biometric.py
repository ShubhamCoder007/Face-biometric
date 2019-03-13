from tkinter import *
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from glob import glob
import os
import cv2

class Biometric:
	
	def __init__(self, window):
		#Initializing the GUI
		self.window = window
		self.window.wm_title("Face Biometric")
		
		l1 = Label(window, text = "Enter the roll no.")
		l1.grid(row = 0, column = 0)
		
		self.roll = StringVar()
		self.e1 = Entry(window, textvariable = self.roll)
		self.e1.grid(row = 0, column = 2, rowspan = 2)
		
		b1 = Button(window, text = "Create profile", width = 12, command = self.create_directory)
		b1.grid(row = 4, column = 4)
		
		b2 = Button(window, text = "Capture face", width = 12, command = self.capture_face)
		b2.grid(row = 5, column = 4)
		
		b3 = Button(window, text = "Train", width = 12, command = self.train)
		b3.grid(row = 6, column = 4)
		
		b4 = Button(window, text = "load model", width = 12, command = self.load)
		b4.grid(row = 7, column = 4)
		
		b5 = Button(window, text = "Close", width = 12, command = self.window.destroy)
		b5.grid(row = 8, column = 4)
		
		b6 = Button(window, text = "Creator", width = 12, command = self.creator)
		b6.grid(row = 8, column = 1)
		
		b7 = Button(window, text = "Detect", width = 12, command = self.detect)
		b7.grid(row = 8, column = 2)
		
		#b8 = Button(window, text = "Creator", width = 12, command = self.creator)
		#b8.grid(row = 8, column = 3)
		
		self.list1 = Listbox(window, height = 3, width = 25)
		self.list1.grid(row = 6, column = 1, columnspan = 3)
		
		#Neural Network Structure initialization
		# Initializing the CNN
		self.classifier = Sequential()

		# Step 1 - Convolution
		self.classifier.add(Conv2D(32, (3, 3), input_shape = (256, 256, 3), activation = 'relu'))

		# Step 2 - Pooling
		self.classifier.add(MaxPooling2D(pool_size = (2, 2)))

		# Adding a second convolutional layer
		self.classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
		self.classifier.add(MaxPooling2D(pool_size = (2, 2)))

		# Step 3 - Flattening
		self.classifier.add(Flatten())

		# Step 4 - Full connection
		self.classifier.add(Dense(units = 128, activation = 'relu'))
		self.classifier.add(Dropout(0.1))
		self.classifier.add(Dense(units = 128, activation = 'relu'))
		self.classifier.add(Dropout(0.1))
		self.classifier.add(Dense(units = self.compute_files(), activation = 'sigmoid'))

		# Compiling the CNN
		self.classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

		
		
		
	def create_directory(self):
		os.mkdir('dataset/training_set/' + self.roll.get())
		os.mkdir('dataset/test_set/' + self.roll.get())
		
		
	def capture_face(self):
		video = cv2.VideoCapture(0)
		frame_no = 0
		print("Capturing ")
		while True:
			
			_,frame = video.read()		#1st is the boolean, 2nd array of image

			cv2.imshow("Capturing",frame)
			key = cv2.waitKey(1)
			
			if frame_no == 200:
				break
				
			if frame_no % 10 == 8 or frame_no % 10 == 9:
				cv2.imwrite('dataset/' + '/test_set/' + self.roll.get()+'/' + str(frame_no)+'.jpg', frame)
			else:
				cv2.imwrite('dataset/' + '/training_set/' +self.roll.get()+'/'+str(frame_no)+'.jpg', frame)
			
			frame_no = frame_no + 1

		self.list1.delete(0,END)
		self.list1.insert(END, "Frames captured:"+str(frame_no))
		video.release()
		cv2.destroyAllWindows()
	
	
	def load(self):
		self.classifier.load_weights('weight_catdog.hdf5')
	
	def save(self):
		self.classifier.save_weights('weight_students.hdf5', overwrite = True)
	
	
	def train(self):
		train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

		test_datagen = ImageDataGenerator(rescale = 1./255)

		training_set = train_datagen.flow_from_directory('dataset/training_set',
														 target_size = (256, 256),
														 batch_size = 32,
														 class_mode = 'sparse')

		test_set = test_datagen.flow_from_directory('dataset/test_set',
													target_size = (256, 256),
													batch_size = 32,
													class_mode = 'sparse')

		#with tf.device('/CPU:2'):
		self.classifier.fit_generator(training_set,
								 steps_per_epoch = self.compute_files()*160,
								 epochs = 10,
								 validation_data = test_set,
								 validation_steps = self.compute_files()*40)
								 
								 
		save()						 
    
		
		
		
		
		
		
	def detect(self):
		video = cv2.VideoCapture(0)

		_, test_img = video.read()
	
		test_img = image.load_img(test_img, target_size = (256,256))
		test_img = image.img_to_array(test_img)
		test_img = np.expand_dims(test_img, axis = 0)
		result.append(classifier.predict(test_img))
	
		result = classifier.predict(test_img)
	
		categories = training_set.class_indices
		value = categories.values()
		key = categories.keys()
		c = {}
		for i,j in zip(value,key):
			c[i]=j

		self.list1.delete(0,END)
		self.list1.append(END,c[int(r)])
	
	
	
	def compute_files(self):
		cwd = os.getcwd()
		os.chdir(cwd + '\\dataset')
		l = os.listdir()
		os.chdir(cwd)
		return len(l)
	
	def creator(self):
		self.list1.delete(0,END)
		self.list1.insert(END, "Created by Shubham Banerjee!")
	
	
	
		
window = Tk()
Biometric(window)
window.mainloop()
