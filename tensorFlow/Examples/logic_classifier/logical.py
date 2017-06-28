from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tflearn

# Logical NOT operator
X = [[0.], [1.]]
Y = [[1.], [0.]]

# Graph definition
with tf.Graph().as_default():
    g = tflearn.input_data(shape=[None, 1])
    g = tflearn.fully_connected(g, 128, activation='linear')
    g = tflearn.fully_connected(g, 128, activation='linear')
    g = tflearn.fully_connected(g, 1, activation='sigmoid')
    g = tflearn.regression(g, optimizer='sgd', learning_rate=2.,
                           loss='mean_square')

    # Model training
    m = tflearn.DNN(g)
    m.fit(X, Y, n_epoch=100, snapshot_epoch=False)

    # Test model
    print("Testing NOT operator")
    print("NOT 0:", m.predict([[0.]]))
    print("NOT 1:", m.predict([[1.]]))

# Logical OR operator
X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
Y = [[0.], [1.], [1.], [1.]]

# Graph definition
with tf.Graph().as_default():
    g = tflearn.input_data(shape=[None, 2])
    g = tflearn.fully_connected(g, 128, activation='linear')
    g = tflearn.fully_connected(g, 128, activation='linear')
    g = tflearn.fully_connected(g, 1, activation='sigmoid')
    g = tflearn.regression(g, optimizer='sgd', learning_rate=2.,
                           loss='mean_square')

    # Model training
    m = tflearn.DNN(g)
    m.fit(X, Y, n_epoch=100, snapshot_epoch=False)
    model = tflearn.DNN(g, tensorboard_verbose=3)

    # Test model
    print("Testing OR operator")
    print("0 or 0:", m.predict([[0., 0.]]))
    print("0 or 1:", m.predict([[0., 1.]]))
    print("1 or 0:", m.predict([[1., 0.]]))
    print("1 or 1:", m.predict([[1., 1.]]))

