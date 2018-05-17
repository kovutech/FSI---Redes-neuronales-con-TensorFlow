import cPickle
import gzip

import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]

neuronsNumber = 10


def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def write_file(string):
    f = open("mnist-history.txt", "a")
    f.write(string)
    f.close()


def chart(a,b):
    plot.title('Practice nn_mnist - Validation error')
    plot.ylabel('Errors')
    plot.xlabel('Epoch')
    valid_handle, = plot.plot(a)
    plot.legend(handles=[valid_handle],
                labels=['Validation error'])
    plot.savefig('./charts/Grafica_mnist-' + str(neuronsNumber) + '.png')
    plot.show()


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
train_y = one_hot(train_y.astype(int), 10)
valid_x, valid_y = valid_set
valid_y = one_hot(valid_y.astype(int), 10)
test_x, test_y = test_set
test_y = one_hot(test_y.astype(int), 10)

# ---------------- Visualizing some element of the MNIST dataset --------------

#import matplotlib.cm as cm
#import matplotlib.pyplot as plt

#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print train_y[57]


# TODO: the neural net!!
x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, neuronsNumber)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(neuronsNumber)) * 0.1)

W1h = tf.Variable(np.float32(np.random.rand(neuronsNumber, neuronsNumber)) * 0.1)
b1h = tf.Variable(np.float32(np.random.rand(neuronsNumber)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(neuronsNumber, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
#h = tf.matmul(x, W1) + b1  # Try this!
h2 = tf.matmul(h, W1h) + b1h
y = tf.nn.softmax(tf.matmul(h2, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
lastError = 0.0
currentError = 10000.0
errorCheck = 1.0;
epoch = 0
validErrorList= []
trainErrorList= []

while errorCheck > 0.0001:
    #Train
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    trainData = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    trainErrorList.append(trainData)

    #Valid error
    lastError = currentError
    currentError = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    errorCheck = lastError - currentError
    validErrorList.append(currentError)

    print "Epoch #:", epoch, "Error: ", currentError, "Difference: ", errorCheck
    epoch += 1
    result = sess.run(y, feed_dict={x: valid_x})
    #for b, r in zip(y_data_valid, result):
    #   print b, "-->", r
    print "----------------------------------------------------------------------------------"

error = 0.0
total = 0.0
result = sess.run(y, feed_dict={x: test_x})
for b, r in zip(test_y, result):
    if np.argmax(b) != np.argmax(r):
        error += 1
    #print b, "-->", r
    total += 1
finalString = 'Neurons:' + str(neuronsNumber) + ' || Epochs:' + str(epoch) + ' || Error: ' + str(round((error / total) * 100,2)) + '%\n'
write_file(finalString)
print finalString

#Chart
chart(validErrorList, trainErrorList)

