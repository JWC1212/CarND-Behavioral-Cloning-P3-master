#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a normalization layer, five convolution layers, and three fully connected layers.This architecture follows NVIDIA's model(see the paper "End to End Learning for Self-Driving Cars").The first three convolutional layers use 5x5 kernel sizes and followed by 2x2 max pooling kernel size. The second two convolutional layers have no max pooling layers just kernel size of 3x3. 

All three fully connected layers have 100, 50, 10 units respectively and each followed by RELU layers to introduce nonlinearity. The data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25). The batch_size is increased to 128 from 32 to raise randomness of data set. The number of epoch is also adjusted to 30 from 10.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road as well recovering from the left and right sides of the road, reverse driving track and driving smoothly along curve. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use convolutional neutral network to map center camera's picture to sheering's angle.

My first step was to use a convolutional neural network model with 2 convolutional layers and 2 fully connected layers. I thought this model might be appropriate because convolution neural network can recognize pictures from camera very well.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set. This implied that the model was underfitting. 

To combat the underfitting, I added another 3 convolutional layers. The first three convolutional layers have each a max pooling layer. I also added one convolutional layers with 10 units.

To combat the overfitting, I modified the model so that the gap of mean squared error between train and test data set.

Then I add a drop-up layer after each fully connected layer. 

The final step was to run the simulator to see how well the car was driving around track one. At first turning-left curve spot where the vehicle fell off the track then recovered to improve the driving behavior in these cases, I recorded these cases and used them to train the model. In such a way the model can be generalized better.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the 5 convolutional layers and each layer's size in order is 16, 32, 48, 64, 64, 3 fully connected layers have each 100, 50 ,10 units and followed by drop-oup layer.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from center :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would increase performance. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had 14611 data points. I then preprocessed this data by extracting only center camera images and normalizing them.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 30 as evidenced by accuracy. I used an adam optimizer so that manually training the learning rate wasn't necessary.
