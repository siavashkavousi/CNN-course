import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # Conv relu layer
        # Input size: (num_train, channels, height, width)
        # Weights size: (num_filters, channels, filter_size, filter_size)
        # Biases size: (num_filters)
        # Output size: (num_train, num_filters, h_nn, w_nn)
        channels, height, width = input_dim
        s = 1
        p = (filter_size - 1) / 2
        h_nn = 1 + (height - filter_size + 2 * p) / s
        w_nn = 1 + (width - filter_size + 2 * p) / s
        # Initializes weights and biases
        self.params['W1'] = np.random.normal(size=(num_filters, channels, filter_size, filter_size), scale=weight_scale)
        self.params['b1'] = np.zeros(num_filters)

        # Max pooling layer
        # Input size: (num_train, num_filters, h_nn, w_nn)
        # Output size: (num_train, num_filters, h_p, w_p)
        pool_width = 2
        pool_height = 2
        pool_stride = 2
        h_p = 1 + (h_nn - pool_height) / pool_stride
        w_p = 1 + (w_nn - pool_width) / pool_stride

        # Affine layer (FC layer)
        # Input size: (num_train, num_filters, h_p, w_p)
        # Weights size: (num_filters * h_p * w_p, hidden_dim)
        # Biases size: (hidden_dim)
        # Output size: (num_train, hidden_dim)
        self.params['W2'] = np.random.normal(size=(num_filters*h_p*w_p, hidden_dim), scale=weight_scale)
        self.params['b2'] = np.zeros(hidden_dim)

        # Affine Layer
        # Input size: (num_train, hidden_dim)
        # Weights size: (hidden_dim, num_classes)
        # Biases size: (num_classes)
        # Output size: (num_train, num_classes)
        self.params['W3'] = np.random.normal(size=(hidden_dim, num_classes), scale=weight_scale)
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


pass
