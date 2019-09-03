# **Behavioral Cloning** 

## ReadMe

### This project is about using the simulator to collect data of good driving behavior and then train a nueral net work to predict the steering angle from images.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depth between 24 and 48.(model.py lines 58-60)
3x3 filter sizes and depths 64 (model.py lines 61-62) 

The model includes RELU layers to introduce nonlinearity (code line 58-62,66,69), and the data is normalized in the model using a Keras lambda layer (code line 56). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 65,68). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 15-38,76). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 75).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 
First it is a little bit hard for me to go through the whole track, maybe you need to try several times to finish a whole round.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was (from easy to complicated)
First, I build a super simple linear model, in order to run the whole process. Of course, the car crashed.

My second step was to use a convolution neural network model similar to the Lenet network ... I thought this model might be appropriate because it used 3 convolutional layers and 3 full connected layers. It turns out that the car could drive for a long distance, but for some point , it will also crashed, especially when the corner is sharp. 
I konw that the model was overfitting, and inorder to beat overfitting, I used the dropout, I used the images from both 3 views of the cameras and I also fliped the images to realize the data augmentation.

I also changed my model to the Nvidia end-to-end selfdriving car model, and I add some activation layer of relu and several dropout layers. I control the number of epochs to prevent model overfitting. finnaly ,I choose 2 epochs and control the loss to about 0.055.


The final step was to run the simulator to see how well the car was driving around track one. It seems that the car could finish one track and the result seems good.


#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes
layer1 : (Convolution2D(24,5,5,subsample = (2,2),activation = "relu"))
layer2:  (Convolution2D(36,5,5,subsample = (2,2),activation = "relu"))
layer3:  (Convolution2D(48,5,5,subsample = (2,2),activation = "relu"))
layer4:  (Convolution2D(64,3,3,activation = "relu"))
layer5： (Convolution2D(64,3,3,activation = "relu"))
layer6:  (Dense(100))
layer7:  (Dense(50))
layer8:  (Dense(10))
layer9:   (Dense(1))
During the dese layer (layer6-ayer9), I also add the activation and dropout.


I tried use a model.summary()to show the info of each layer after I finish all the process and also generate the output video.
However some file on the workspace is deleted by me and it seems the script could not run normally.
#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 

Then I use the code to add the left and right view picture of the camera to my training set

I also used the code to flip the image and add it to the training set. You can refer the code below:
```
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
# use flip to do the data augmentation
augmented_images, augmented_measurements = [],[]
for image,measurement in zip (images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*(-1.0))
```

After the collection process,  I finished the data augmentation and add the data to my training data set.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by the validation loss change. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Finally I got a not bad result and the car could run automatically for a full round.
