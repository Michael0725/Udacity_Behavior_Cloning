import csv
import cv2
import scipy.ndimage as ndimage
import numpy as np



lines= []
with open('data/driving_log.csv') as csvfile:
    reader =csv.reader(csvfile)
    for line in reader:
        lines.append(line)
angle_correct = 0.2
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename =source_path.split('\\')[-1]
        current_path = 'data/IMG/'+filename
        image = ndimage.imread(current_path)
        images.append(image)
        if i ==0:
            measurement = float(line[3])
        if i ==1:
            measurement = float(line[3])+angle_correct
        if i ==2:
            measurement =float(line[3])-angle_correct
        measurements.append(measurement)
  
augmented_images, augmented_measurements = [],[]
for image,measurement in zip (images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*(-1.0))
    
                            
                        
    

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Lambda,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
model.add(Lambda(lambda x:x/255.0 - 0.5,input_shape =(160,320,3)))
model.add(Cropping2D(cropping =((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(36,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(48,5,5,subsample = (2,2),activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse',optimizer = 'adam')
model.fit(X_train,y_train,validation_split = 0.2,shuffle = True,nb_epoch =3)
model.save('model.h5')
exit()




    
    
    