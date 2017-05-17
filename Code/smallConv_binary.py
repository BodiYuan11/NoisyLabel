# Import MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
mnistfake=mnist

# Initials
fakerate=0.05
dataSize=55000
batchSize=128
validateRate=100
maxInteration=200000

# Get binary dataset
trainData=mnistfake.train.next_batch(dataSize)[0]
trainLabel=mnistfake.train.next_batch(dataSize)[1]
label1=4
label2=6
index1=np.where(np.nonzero(trainLabel)[1]==label1)[0]
index2=np.where(np.nonzero(trainLabel)[1]==label2)[0]
trainData1=trainData[index1]
trainLabel1=np.zeros([np.size(index1),2])
trainLabel1[:,0]=1
trainData2=trainData[index2]
trainLabel2=np.zeros([np.size(index2),2])
trainLabel2[:,1]=1
trainData=np.append(trainData1,trainData2,0)
trainLabel=np.append(trainLabel1,trainLabel2,0)

testData=np.copy(trainData)
testLabel=np.copy(trainLabel)

# Fake the dataset
import random as rd
n,d=np.shape(trainLabel)


for i in rd.sample(range(n),int(n*fakerate)):
	trainLabel[i]=[not(trainLabel[i][0]),not(trainLabel[i][1])]

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

W = tf.Variable(tf.zeros([784,2]))
b = tf.Variable(tf.zeros([2]))
sess.run(tf.global_variables_initializer())

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

open("accuracy0.05", 'w').close()

for i in range(maxInteration):
	#print i
	#batchfake = mnistfake.train.next_batch(batchSize)
	batchIndex=rd.sample(range(n),batchSize)
	batchData=trainData[batchIndex]
	batchLabel=trainLabel[batchIndex]
	if i%validateRate == 0:
		validate_accuracy = accuracy.eval(feed_dict={x:testData, y_: testLabel, keep_prob: 1.0})
		#validate_accuracy = accuracy.eval(feed_dict={x: batchfake[0], y_: batchfake[1], keep_prob: 1.0})
		print("step %d, validation accuracy %g"%(i, validate_accuracy))
		with open("accuracy0.05","a") as text_file:
			text_file.write("%g\n"%validate_accuracy)
	#train_step.run(feed_dict={x: batchfake[0], y_: batchfake[1], keep_prob: 0.5})
	train_step.run(feed_dict={x: batchData, y_: batchLabel, keep_prob: 1})

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))




