import numpy as np
import tensorflow as tf
import pickle
import struct
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)

# # start tensorflow interactiveSession
# import tensorflow as tf
# sess = tf.InteractiveSession()
#
# # weight initialization
# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape = shape)
#     return tf.Variable(initial)
#
# # convolution
# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# # pooling
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
# # Create the model
# # placeholder
# x = tf.placeholder("float", [None, 784])
# y_ = tf.placeholder("float", [None, 10])
# # variables
# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))
#
# y = tf.nn.softmax(tf.matmul(x,W) + b)
#
# # first convolutinal layer
# w_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])
#
# x_image = tf.reshape(x, [-1, 28, 28, 1])
#
# h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)
#
# # second convolutional layer
# w_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
#
# h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
#
# # densely connected layer
# w_fc1 = weight_variable([7*7*64, 1024])
# b_fc1 = bias_variable([1024])
#
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
#
# # dropout
# keep_prob = tf.placeholder("float")
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# # readout layer
# w_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])
#
# y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
#
# # train and evaluate the model
# cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# sess.run(tf.initialize_all_variables())
# for i in range(20000):
#     batch = mnist.train.next_batch(50)
#     if i%100 == 0:
#         train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
#         print "step %d, train accuracy %g" %(i, train_accuracy)
#     train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
#
# print "test accuracy %g" % accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})

#
#
# class Data:
#     def __init__(self):
#
#         self.K = 10
#         self.N = 60000
#         self.M = 10000
#         self.BATCHSIZE = 2000
#         self.reg_factor = 1e-3
#         self.stepsize = 1e-2
#         self.train_img_list = np.zeros(( self.N, 28*28 ))
#         self.train_label_list = np.zeros((self.N, 1))
#
#
#
#         self.test_img_list = np.zeros(( self.M, 28*28 ))
#         self.test_label_list = np.zeros((self.M, 1))
#
#         self.sess = tf.InteractiveSession()
#         self.sess.run( tf.initialize_all_variables() )
#         self.loss_list = []
#         self.init_network()
#
#         self.read_train_images( 'train-images-idx3-ubyte')
#         self.read_train_labels( 'train-labels-idx1-ubyte')
#
#         self.train_data = np.append( self.train_img_list, self.train_label_list, axis = 1 )
#
#
#         self.read_test_images('t10k-images-idx3-ubyte')
#         self.read_test_labels('t10k-labels-idx1-ubyte')
#
#
#
#
#
#
#     def GetOneHot(self, transfer_list):
#         const_zero = np.zeros( [ transfer_list.shape[0], 10 ] )
#
#         for i in range( transfer_list.shape[0] ):
#             const_zero[ i ][ int(transfer_list[ i ]) ] = 1
#
#         return const_zero

    # def get_weights(self, shape):
    #     weight = tf.truncated_normal(shape, stddev=0.01)
    #     return tf.Variable(weight)
    #
    #
    # # def GetWeights(self,shape):
    # #     weights = tf.truncated_normal( shape, stddev= 0.1 )
    # #     return tf.Variable( weights )
    #
    # def get_bias(self, shape):
    #     bias = tf.constant(0.01, shape=shape)
    #     return tf.Variable(bias)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable( shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



        # np.random.shuffle(self.train_data)
        # img_list = self.train_data[:self.BATCHSIZE, :-1]
        # label_list = self.train_data[:self.BATCHSIZE, -1:]
        # one_hot_label = self.GetOneHot(label_list)
        # print one_hot_label.shape
        # # print one_hot_label
        # self.optimizer.run(
        #     feed_dict = {
        #         self.input_layer: img_list, self.label: one_hot_label
        #     }
        # )
        # # self.sess.run( self.optimizer, feed_dict = { } )
        # print i, self.cost
    return


sess = tf.InteractiveSession()



x = tf.placeholder("float", [None, 28 * 28])
y_ = tf.placeholder("float", [None, 10])

w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# train and evaluate the model
cross_entropy = -tf.reduce_sum(tf.log(y_conv))
optimizer = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)
sess.run(tf.initialize_all_variables())
for i in range(1000):
    print i
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={self.input_layer:batch[0], self.label:batch[1], keep_prob:1.0})
        print "step %d, train accuracy %g" %(i, train_accuracy)
    optimizer.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # self.label = tf.placeholder( "float", [None, self.K] )
    # self.input_layer = tf.placeholder( "float", [ None, 28*28 ] )
    # self.input = tf.reshape( self.input_layer, ( -1, 28, 28, 1 ) )
    # W1 = self.get_weights( [ 5,5,1,32 ] )
    # b1 = self.get_bias( [ 32 ] )
    #
    # W2 = self.get_weights( [ 5,5,32,64 ] )
    # b2 = self.get_bias( [ 64 ] )
    #
    # W_fc1 = self.get_weights( [ 7 * 7 * 64, 1024 ] )
    # b_fc1 = self.get_bias( [ 1024 ] )
    #
    # W_fc2 = self.get_weights( [ 1024, 10 ] )
    # b_fc2 = self.get_bias([ 10 ] )
    #
    # h1 = tf.nn.relu(
    #     tf.nn.conv2d( self.input, W1, strides= [ 1,1,1,1 ], padding= 'SAME' )
    #     + b1)
    # p1 = tf.nn.max_pool( h1, [ 1,2,2,1 ],strides= [1,2,2,1],padding= 'SAME' )
    #
    #
    # h2 = tf.nn.relu(
    #     tf.nn.conv2d( p1, W2, strides = [ 1,1,1,1 ], padding = 'SAME')
    #     + b2
    # )
    # p2 = tf.nn.max_pool( h2, [1,2,2,1], strides=[1,2,2,1], padding= 'SAME' )
    #
    # p2_flatten = tf.reshape( p2, [ -1, 7 * 7 * 64 ] )
    #
    # conv1 = tf.nn.relu(tf.matmul( p2_flatten, W_fc1 ) + b_fc1)
    #
    # output = tf.nn.softmax( tf.matmul( conv1, W_fc2 ) + b_fc2 )
    #
    # cross_entropy = -tf.reduce_sum( self.label * tf.log(output))
    # self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # self.cost = -1 * tf.reduce_sum(  tf.log(tf.mul(output, self.label)) )
    # self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.cost)

def predict(self):
    test_label_tensor = tf.Variable( tf.constant( self.test_label_list ) )

    prediction_cross_entropy = self.output.eval( feed_dict = {
        self.input_layer : self.test_img_list
    } )

    prediction = tf.argmax(prediction_cross_entropy,1)
    correct_prediction = tf.equal( prediction, test_label_tensor )
    accuracy = tf.reduce_mean( tf.cast( correct_prediction ), "float" )
    print 'final accuracy: ', accuracy


    # def read_train_images(self,filename):
    #     binfile = open(filename, 'rb')
    #     buf = binfile.read()
    #     index = 0
    #     magic, self.train_img_num, self.numRows, self.numColums = struct.unpack_from('>IIII', buf, index)
    #     print magic, ' ', self.train_img_num, ' ', self.numRows, ' ', self.numColums
    #     index += struct.calcsize('>IIII')
    #     for i in range(self.train_img_num):
    #         im = struct.unpack_from('>784B', buf, index)
    #         index += struct.calcsize('>784B')
    #         im = np.array(im)
    #         im = im.reshape(1, 28 * 28 )
    #         self.train_img_list[ i , : ] = im
    #
    #
    # def read_train_labels(self,filename):
    #     binfile = open(filename, 'rb')
    #     index = 0
    #     buf = binfile.read()
    #     binfile.close()
    #
    #     magic, self.train_label_num = struct.unpack_from('>II', buf, index)
    #     index += struct.calcsize('>II')
    #
    #     for i in range(self.train_label_num):
    #         # for x in xrange(2000):
    #         label_item = int(struct.unpack_from('>B', buf, index)[0])
    #         self.train_label_list[ i , : ] = label_item
    #         index += struct.calcsize('>B')
    #
    # def read_test_images(self, filename):
    #     binfile = open(filename, 'rb')
    #     buf = binfile.read()
    #     index = 0
    #     magic, self.test_img_num, self.numRows, self.numColums = struct.unpack_from('>IIII', buf, index)
    #     print magic, ' ', self.test_img_num, ' ', self.numRows, ' ', self.numColums
    #     index += struct.calcsize('>IIII')
    #     for i in range(self.test_img_num):
    #         im = struct.unpack_from('>784B', buf, index)
    #         index += struct.calcsize('>784B')
    #         im = np.array(im)
    #         im = im.reshape(1, 28*28)
    #         self.test_img_list[i, :] = im
    # def read_test_labels(self,filename):
    #     binfile = open(filename, 'rb')
    #     index = 0
    #     buf = binfile.read()
    #     binfile.close()
    #
    #     magic, self.test_label_num = struct.unpack_from('>II', buf, index)
    #     index += struct.calcsize('>II')
    #
    #     for i in range(self.test_label_num):
    #         # for x in xrange(2000):
    #         label_item = int(struct.unpack_from('>B', buf, index)[0])
    #         self.test_label_list[i, :] = label_item
    #         index += struct.calcsize('>B')
#
# def main():
#     data = Data()
#     data.train()
#     data.predict()
#
#
# if __name__ == '__main__':
#     main()