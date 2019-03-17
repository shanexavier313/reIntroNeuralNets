# reIntroNeuralNets
Code used in our Re-Introduction to Neural Networks articles on dimensionless.tech

Started with building a single perceptron for binary out-puts. 

Then we built a more complex net with hiden layers to model non-linear probabilities. 

Finally we evolved the network to enable multi-class Classification

This project was modeled on the articles by [Usman Malik](https://stackabuse.com/creating-a-neural-network-from-scratch-in-python-multi-class-classification/)

## Optimization 
When testing the binary classification neural network, it quickly became obvious that the results varied significantly depending on the random seed. A random seed of 0 resulted in a relatively low ~1.5 MSE. But many other seeds (50,75,100) where significantly higher between ~4-5 MSE. 

It was clear that we needed to optimize the model. To maintain simplicity, we focused on tuning the learning rate. In future exercises we will work on tuning the number of hidden layers and the number of units in each of those layers.  
