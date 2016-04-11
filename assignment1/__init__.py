import data_utils
from classifiers.linear_classifier import *
from classifiers.k_nearest_neighbor import *

X, y = data_utils.load_CIFAR_batch('datasets\cifar-10\data_batch_1')
Xte, yte = data_utils.load_CIFAR_batch('datasets\cifar-10\\test_batch')
# X, y, Xte, yte = data_utils.load_CIFAR10('datasets\cifar-10')

X = X.reshape(X.shape[0], 32 * 32 * 3)
Xte = Xte.reshape(Xte.shape[0], 32 * 32 * 3)

# Data Pre-processing by zero-centering and adding an extra dimension 1 to each sample
num_train, dim = X.shape
mean = np.mean(X, axis=0)
X -= mean
Xte -= mean
X = np.insert(X, dim, 1, axis=1)
Xte = np.insert(Xte, dim, 1, axis=1)

# Nearest Neighbor (NOTE: data pre-processing has no effect on NN)
# nn = KNearestNeighbor()
# nn.train(X, y)
# y_pred = nn.predict(Xte)

# LinearSVM
lc = LinearSVM()
lc.train(X, y, batch_size=200, num_iters=1)
y_pred = lc.predict(Xte)

print 'accuracy: %f' % (np.mean(y_pred == yte))
