import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    num_classes = W.shape[1]
    num_train, num_dim = X.shape
    dW = np.zeros_like(W)

    for i in xrange(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)

        p = np.exp(scores) / np.sum(np.exp(scores))
        loss += -np.log(p[y[i]])

        p[y[i]] -= 1
        dW -= X[i].reshape(num_dim, 1) * p.reshape(1, num_classes)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    num_train, num_dim = X.shape

    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)

    p = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    loss = -np.sum(np.log(p[range(num_train), y]))

    p[range(num_train), y] -= 1
    dW = X.T.dot(p)

    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW
