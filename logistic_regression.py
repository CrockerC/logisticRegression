import collections
import math
import numpy as np
import string
import re


class Logistic_Regression:
    def __init__(self):
        self.W = None
        self.b = None

    def fit(self, x, y, batch_size=64, iteration=2000, learning_rate=1e-2):
        """
        Train this Logistic Regression classifier using mini-batch stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - iteration: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        
        Use the given learning_rate, iteration, or batch_size for this homework problem.

        Returns:
        None
        """
        dim = x.shape[1]
        num_train = x.shape[0]

        # initialize W
        if self.W is None:
            self.W = 0.001 * np.random.randn(1, dim)
            self.b = 0

        for it in range(iteration):
            batch_ind = np.random.choice(num_train, batch_size)

            x_batch = x[batch_ind]
            y_batch = y[batch_ind]
            
            # Calculate loss and update W, b

            y_pred = self.predict(x_batch)
            loss, gradient = self.loss(x_batch, y_pred, y_batch)

            self.W = self.W - (learning_rate * gradient['dW'])
            self.b = self.b - (learning_rate * gradient['db'])

            corr = 0
            for i in range(y_pred.shape[0]):
                if y_pred[i] == y_batch[i]:
                    corr += 1

            acc = corr / (y_pred.shape[0])

            if it % 50 == 0:
                print('iteration %d / %d: accuracy : %f: loss : %f' % (it, iteration, acc, loss))

    def predict(self, x):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.
        Inputs:

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """

        y_pred = self.sigmoid(np.dot(x, self.W.T) + self.b)
        y_pred = np.rint(y_pred)

        return y_pred

    def loss(self, x_batch, y_pred, y_batch):
        """
        Compute the loss function and its derivative. 
        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        Returns: A tuple containing:
        - loss as a single float
        - gradient dictionary with two keys : 'dW' and 'db'
        """
        gradient = {'dW': None, 'db': None}

        # todo, is this correct? 90% sure yes
        n = x_batch.shape[0]
        loss = -(1 / n) * np.sum(-y_batch * np.log(y_pred + 1e-6) - (1 - y_batch) * np.log(1 - y_pred + 1e-6))

        # todo, is this right?
        gradient.update({'dW': (1 / n) * np.dot(x_batch.T, (y_pred - y_batch)).T})
        gradient.update({'db': (1 / n) * (np.sum(y_pred - y_batch))})

        return loss, gradient

    def sigmoid(self, z):
        """
        Compute the sigmoid of z
        Inputs:
        z : A scalar or numpy array of any size.
        Return:
        s : sigmoid of input
        """
        z = np.clip(z, -500, 500)
        s = 1 / (1 + np.exp(-z + 1e-6))

        return s
