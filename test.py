# import matplotlib
# matplotlib.use('TkAgg')  #for mac using matplotlib
# # This should be done before `import matplotlib.pyplot`
# # 'Qt4Agg' for PyQt4 or PySide, 'Qt5Agg' for PyQt5
# import matplotlib.pyplot as plt
# import numpy as np
#
# print ('try')
#
# t = np.linspace(0, 20, 500)
# plt.plot(t, np.sin(t))
# plt.show()
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

train_y=[1,1,0,2]
print(len(np.unique(train_y)))

# failed

# x = pickle.load(open('dataset/cifar_100_features.p', 'rb'))
# y = pickle.load(open('dataset/cifar_100_labels.p', 'rb'))
# x = np.array(x, dtype=np.float32)
# y = np.array(y, dtype=np.int32)
# print(x.shape, y.shape)
#
# x, train_x, y, train_y = train_test_split(x, y, test_size=int(1e4), stratify=y)
#
# print("Training set size:  X: {}, y: {}".format(train_x.shape, train_y.shape))
# print(x[0])
# print(train_x[0])
# print(y[0])
# print(train_y[0])
# print("here")

def different_return(x):

    if x>1:
        return x
    else:
        return x, (x, x+1)

a, aux = different_return(1)
print(a, aux)
print(10*'*')
a = different_return(2)
print(a)


import numpy as np
X, y = np.arange(10).reshape((5, 2)), range(5)
print(X)
print(list(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1)

print(X_train)
print(X_test)
print(y_train)
print(y_test)


def print_arg(arg):
    l2_ratio=args.target_l2_ratio
    print(l2_ratio)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_l2_ratio', type=float, default=1e-8)
    args = parser.parse_args()
    print(vars(args))

    print_arg(args)

