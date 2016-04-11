import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin

                dW[:, j] = X[i]
                dW[:, y[i]] -= dW[:, j]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    num_train = X.shape[0]
    num_classes = W.shape[1]
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    margins = np.zeros((num_train, num_classes))
    delta = 1.0

    # Calculate loss function
    scores = X.dot(W)

    for i in xrange(num_train):
        margins[i] = np.maximum(0, scores[i] - scores[i, y[i]] + delta)
        margins[i, y[i]] = 0
    loss = np.sum(margins)

    # Calculate gradient
    weight = np.array((margins > 0).astype(int))
    w_yi = -np.sum(weight, axis=1)
    for i in xrange(num_train):
        weight[i, y[i]] = w_yi[i]

    for i in xrange(num_train):
        dW += X[i].reshape(-1, 1).dot(weight[i].reshape(-1, 1).T)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW
