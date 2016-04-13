import data_utils
from classifiers.linear_classifier import *
from classifiers.k_nearest_neighbor import *
from timeit import default_timer as timer


def run_knn():
    nn = KNearestNeighbor()
    nn.train(X, y)
    y_pred = nn.predict(Xte)
    print 'accuracy: %f' % (np.mean(y_pred == yte))


def run_svm():
    lc = LinearSVM()
    lc.train(X, y, batch_size=200, num_iters=10000)
    y_pred = lc.predict(Xte)
    print 'accuracy: %f' % (np.mean(y_pred == yte))


def run_softmax():
    sm = Softmax()
    sm.train(X, y, num_iters=500)
    y_pred = sm.predict(Xte)
    print 'accuracy: %f' % (np.mean(y_pred == yte))


def measure_elapsed_time(f):
    start = timer()
    f
    end = timer()
    print 'start time: ', start, ' end time: ', end, ' duration: ', end - start


# X, y = data_utils.load_CIFAR_batch('datasets\cifar-10\data_batch_1')
# Xte, yte = data_utils.load_CIFAR_batch('datasets\cifar-10\\test_batch')
X, y, Xte, yte = data_utils.load_CIFAR10('datasets\cifar-10')

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
measure_elapsed_time(run_knn())
# LinearSVM
measure_elapsed_time(run_svm())
# Softmax
measure_elapsed_time(run_softmax())
