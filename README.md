# reIntroNeuralNets
Code used in our Re-Introduction to Neural Networks articles on dimensionless.tech

Started with building a single perceptron for binary out-puts. 

Then we built a more complex net with hiden layers to model non-linear probabilities. 

Finally we evolved the network to enable multi-class Classification

This project was modeled on the articles by [Usman Malik](https://stackabuse.com/creating-a-neural-network-from-scratch-in-python-multi-class-classification/)

## Optimization 
When testing the binary classification neural network, it quickly became obvious that the results varied significantly depending on the random seed. A random seed of 0 resulted in a relatively low ~1.5 MSE. But many other seeds (50,75,100) where significantly higher between ~4-5 MSE. 

It was clear that we needed to optimize the model. To maintain simplicity, we focused on tuning the learning rate. In future exercises we will work on tuning the number of hidden layers and the number of units in each of those layers. 

## Multi-Class Neural Network 

We also wrote a simple multi-class neural net which can identify if an object is in one of 3 different classes. The data set was manually created with points scattered around origins of different (x,y) values. Unlike the other networks, we used a variety of activation functions and the output layer has 3 nodes. The input nodes uses the sigmoid funciton as its activation function, the same as in our other networks. But the hidden layer nodes uses the softmax function as the activation function for the output nodes. We use the softmax function because it can receive a vector as the input and return a vector of equal length as the output. Since our output can be a variety of classes (1 value from a vector of 3) it makes more sense to use the softmax function.

We also changed our cost function. In the other neural networks we used the Mean Squared Error (MSE) cost function. For the Multi-Class Neural Network we used the cross-entropy function. For multi-class classification problems, this cost function is known to outperform the MSE cost function.  
