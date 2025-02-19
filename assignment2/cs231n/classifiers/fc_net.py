import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecture should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        self.params['W1'] = np.random.normal(size=(input_dim, hidden_dim), scale=weight_scale)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(size=(hidden_dim, num_classes), scale=weight_scale)
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        w1, b1 = self.params['W1'], self.params['b1']
        w2, b2 = self.params['W2'], self.params['b2']

        h1, cache1 = affine_relu_forward(X, w1, b1)
        h2, cache2 = affine_forward(h1, w2, b2)
        scores = h2

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, dout = softmax_loss(scores, y)
        dh2, dw2, db2 = affine_backward(dout, cache2)
        dh1, dw1, db1 = affine_relu_backward(dh2, cache1)

        dw1 += self.reg * w1
        dw2 += self.reg * w2
        grads = {'W1': dw1, 'b1': db1, 'W2': dw2, 'b2': db2}

        loss += 0.5 * self.reg * (np.sum(w1 * w1) + np.sum(w2 * w2))

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        dims = [input_dim] + hidden_dims + [num_classes]

        for i in range(self.num_layers):
            self.params['W' + str(i + 1)] = np.random.normal(size=(dims[i], dims[i + 1]), scale=weight_scale)
            self.params['b' + str(i + 1)] = np.zeros(dims[i + 1])

        # if use_batchnorm:
        #     for i in range(len(dims) - 1):
        #         self.params['gamma' + str(i + 1)] = np.ones(dims[i + 1])
        #         self.params['beta' + str(i + 1)] = np.zeros(dims[i + 1])

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        hidden = {}
        num_layers = self.num_layers

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        # Forward pass
        hidden['h0'] = X
        # Performs dropout on input layer
        if self.use_dropout:
            h, cache = dropout_forward(X, self.dropout_param)
            hidden['h_dropout0'] = h
            hidden['cache_dropout0'] = cache

        for i in range(num_layers):
            idx = i + 1
            w = self.params['W' + str(idx)]
            b = self.params['b' + str(idx)]
            h = hidden['h' + str(idx - 1)]
            if self.use_dropout:
                h = hidden['h_dropout' + str(idx - 1)]

            if idx == num_layers:
                h, cache = affine_forward(h, w, b)
                hidden['h' + str(idx)] = h
                hidden['cache' + str(idx)] = cache
            else:
                h, cache = affine_relu_forward(h, w, b)
                hidden['h' + str(idx)] = h
                hidden['cache' + str(idx)] = cache

                if self.use_dropout:
                    # Performs dropout on the current layer
                    h = hidden['h' + str(idx)]
                    h, cache = dropout_forward(h, self.dropout_param)
                    hidden['h_dropout' + str(idx)] = h
                    hidden['cache_dropout' + str(idx)] = cache

        scores = hidden['h' + str(num_layers)]

        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        loss, dout = softmax_loss(scores, y)
        # Computes loss
        for w in [self.params[f] for f in self.params.keys() if f[0] == 'W']:
            loss += 0.5 * self.reg * np.sum(w * w)

        # Backward pass
        hidden['dh' + str(num_layers)] = dout

        for i in range(num_layers)[::-1]:
            idx = i + 1
            dout = hidden['dh' + str(idx)]
            out_cache = hidden['cache' + str(idx)]

            if idx == num_layers:
                dh, dw, db = affine_backward(dout, out_cache)
                hidden['dh' + str(idx - 1)] = dh
                hidden['dW' + str(idx)] = dw
                hidden['db' + str(idx)] = db
            else:
                if self.use_dropout:
                    out_cache_dropout = hidden['cache_dropout' + str(idx)]
                    dout = dropout_backward(dout, out_cache_dropout)

                dh, dw, db = affine_relu_backward(dout, out_cache)
                hidden['dh' + str(idx - 1)] = dh
                hidden['dW' + str(idx)] = dw
                hidden['db' + str(idx)] = db

        for i in range(1, self.num_layers):
            hidden['dW' + str(i)] += self.reg * self.params['W' + str(i)]

        list_dw = {key[1:]: val for key, val in hidden.iteritems() if key[:2] == 'dW'}
        list_db = {key[1:]: val for key, val in hidden.iteritems() if key[:2] == 'db'}

        grads.update(list_dw)
        grads.update(list_db)

        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
