## Melanoma Detection Assignment

### Introduction
In this assignment, you will create a multiclass classification model using a custom convolutional neural network (CNN) built with TensorFlow.

### Problem Statement 
To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis. The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.


### General Information:
We need build the model for the demand for shared bikes with the available independent variables. It will be used by the management to understand how exactly the demands vary with different features. They can accordingly manipulate the business strategy to meet the demand levels and meet the customer's expectations. Further, the model will be a good way for management to understand the demand dynamics of a new market. 

###Dataset:
The dataset used to train and test the model were extracted from CNN_assignment.zip folder.
Augmentation: The dataset contains 2,357 images of malignant and benign oncological conditions, sourced from the International Skin Imaging Collaboration (ISIC). These images were categorized based on the classification provided by ISIC, and all subsets were divided to contain an equal number of images.
To overcome the issue of class imbalance, used a python package Augmentor (https://augmentor.readthedocs.io/en/master/) to add more samples across all classes so that none of the classes have very few samples.

### Model Design:

- Rescaling Layer: This layer scales input values from the [0, 255] range to the [0, 1] range.
  
- Convolutional Layer: Convolutional layers apply a convolution operation to the input, passing the resulting output to the next layer. A convolution operation reduces the size of the input and aggregates the information within its receptive field into a single value. For example, applying a convolution to an image results in a smaller image where each pixel represents the combined information from its corresponding region.
  
- Pooling Layer: Pooling layers reduce the size of the feature maps, thus decreasing the number of parameters and the computational load of the network. The pooling layer summarizes the features within a specific region of the feature map produced by a convolutional layer.
  
- Dropout Layer: The Dropout layer randomly sets a fraction of input units to zero during each training step, with a probability defined by the rate. This helps to prevent overfitting.
  
- Flatten Layer: Flattening transforms the output of the convolutional layers into a one-dimensional array to be passed to the subsequent layer. The flattened output is then used as input for the final fully-connected layer in the classification model.
  
- Dense Layer: A dense layer in a neural network is fully connected, meaning every neuron in the dense layer receives input from all the neurons of the previous layer.
  
- Activation Function (ReLU): The ReLU (Rectified Linear Unit) activation function is a piecewise linear function that outputs the input directly if it is positive and outputs zero otherwise. ReLU helps to solve the vanishing gradient problem, enabling faster learning and improved model performance.
  
- Activation Function (Softmax): The Softmax function is commonly used in the output layer of neural networks that predict a multinomial probability distribution. Its main benefit is that it ensures the output probabilities range from 0 to 1 and that the sum of all probabilities equals one.


# Conclusion:
The model was trained with ~20 epochs and tested with ~50 epochs and found to be 92% accurate (Refer to the "Train the Model" section in the python note book "Melanoma_Detection_Assignment.ipynb")


## Contact
Created by Vidya S