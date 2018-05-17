import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plot


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
    f = open("iris-history.txt", "a")
    f.write(string)
    f.close()


def chart(a,b):
    plot.ylabel('Errors')
    plot.xlabel('Epoch')
    valid_handle, = plot.plot(a)
    train_handle, = plot.plot(b)
    plot.legend(handles=[valid_handle, train_handle],
                labels=['Validation error', 'Train error'])
    plot.savefig('./charts/Grafica_iris-' + str(neuronsNumber) + '.png')

data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

trainRangeData = int(np.floor(len(x_data) * 0.7))

train_x = x_data[:trainRangeData]
train_y = y_data[:trainRangeData]

validRangeData = trainRangeData + int(np.floor(len(x_data) * 0.15))

valid_x = x_data[trainRangeData:validRangeData]
valid_y = y_data[trainRangeData:validRangeData]

test_x = x_data[validRangeData:]
test_y = y_data[validRangeData:]


print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, neuronsNumber)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(neuronsNumber)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(neuronsNumber, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
epoch = 0
validErrorList= []
trainErrorList= []

for epoch in xrange(150):
    #Train
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    #Valid
    currentError = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    validErrorList.append(currentError)
    print "Epoch #:", epoch, "Error: ", currentError
    result = sess.run(y, feed_dict={x: valid_x})
    for b, r in zip(valid_y, result):
        print b, "-->", r
    print "----------------------------------------------------------------------------------"

#Test
error = 0.0
total = 0.0
result = sess.run(y, feed_dict={x: test_x})
for b, r in zip(test_y, result):
    if np.argmax(b) != np.argmax(r):
        error += 1
        print "Next has error"
    print b, "-->", r
    total += 1
print "----------------------------------------------------------------------------------"
finalString = 'Neurons:' + str(neuronsNumber) + ' || Epochs:' + str(epoch) + ' || Percentage: ' + str(round((error / total) * 100,2)) + '\n'
write_file(finalString)
print finalString

#Chart
chart(validErrorList, trainErrorList)