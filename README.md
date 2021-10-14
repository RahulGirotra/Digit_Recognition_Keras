# Digit_Recognition_Keras


## About:
MNIST dataset from keras is used for Digit Recogintion {0-9}. Convolution Neural Network used for modelling.


* Data-set is loaded from mnist of keras

* Archiecture:-
1) Two convolution layers
2) Two max. pooling layers
3) flatten layer
4) Dense layer
5) Dense Output layer {Will have 10 nodes, because its classifier for 10 nodes}

In convolution 2D layer 'relu' activation function is used. Same in first dense layer.
But in output dense layer 'softmax' activation function is used because its a classification problem of more than one classes.

Model is ran over 60000 random train images and 10000 random test images from the dataset.



>> loss: 0.0834 - accuracy: 0.9819
{Result may vary a little because of randomness of data set used for training of the model}

Note:- Please check the "MNIST_using_RNN.ipynb" for better understanding of the model.



Architecture and trainable parameters are then saved to 'model.h5'. Which is loaded in "app.py" for deployment using Flask. 

