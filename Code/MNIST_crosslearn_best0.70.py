from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import random
import os

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

def train_set(train_data,train_labels,iternum,AccLogFile,ModelDir,Best=True):
    if not os.path.exists(ModelDir):
        os.makedirs(ModelDir)
    open(AccLogFile,'w').close()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    CurBestAcc = 0;
    for i in range(iternum):
      index = random.sample(range(train_data.shape[0]),50)
      batch = (train_data[index],train_labels[index])
      if i%200 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, acc %g"%(i, train_accuracy))
        with open(AccLogFile,"a") as text_file:
            text_file.write("step %d, acc %g\n"%(i,train_accuracy))
      if((not Best) and i%1000 == 0):
        saver.save(sess,ModelDir, global_step=i)
      if i%500 == 0:
        test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        if(Best and test_accuracy>CurBestAcc):
          saver.save(sess,ModelDir, global_step=i)
          CurBestAcc = test_accuracy
        with open(AccLogFile,"a") as text_file:
            text_file.write("step %d, test accuracy %g\n"%(i,test_accuracy))
        print("test accuracy %g"% test_accuracy)
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

def update_labels(data,labels_prev,ModelDir):
    labels = np.zeros(labels_prev.shape)
    sliceNum = 50
    sliceSize = data.shape[0]/sliceNum
    sess.run(tf.global_variables_initializer())
    saver=tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(ModelDir)
    print ckpt.model_checkpoint_path
    saver.restore(sess, ckpt.model_checkpoint_path)
    for i in range(sliceNum):
        labels[i*sliceSize:(i+1)*sliceSize] = sess.run(y_conv,{x:data[i*sliceSize:(i+1)*sliceSize],keep_prob: 1.0})
    correct_prediction = tf.equal(tf.argmax(labels,1), tf.argmax(labels_prev,1))
    acc = sess.run(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
    print acc

    maxIndex = np.argmax(labels,1)
    labels = np.zeros(labels_prev.shape)
    for i in range(labels_prev.shape[0]):
        labels[i,maxIndex[i]]=1
    return labels

def update_cross_shuffle(labels_target,labels_support):
    labels_result = labels_target.copy()
    for i in range(labels_result.shape[0]):
        if np.argmax(labels_result[i,:])!=np.argmax(labels_support[i,:]):
            labels_result[i,:] = np.zeros(10)
            labels_result[i,random.randint(0,9)] = 1
    return labels_result


## shuffle 10 classes
shuffle_pre = 0.70;
tot_len = len(mnist.train.labels)
change_len = int(tot_len*shuffle_pre)
indexlist = random.sample(range(tot_len),change_len)
for curp in indexlist:
  tind=-1
  for j, xx in enumerate(mnist.train.labels[curp]):
    if xx==1:
      tind = j
      break
  mnist.train.labels[curp][tind]=int(0)
  tind+=random.randint(1,9)
  tind%=10
  mnist.train.labels[curp][tind]=int(1)

train_data = mnist.train.images # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
print train_data.shape

data_length = train_data.shape[0]
train_data1 = train_data[0:data_length/2]
train_data2 = train_data[data_length/2:data_length]
train_labels1_iter0 = train_labels[0:data_length/2]
train_labels2_iter0 = train_labels[data_length/2:data_length]
print train_data2.shape

# ConstructGraph:
tf.reset_default_graph()
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#W = tf.Variable(tf.zeros([784,10]))
#b = tf.Variable(tf.zeros([10]))

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

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





# iter0
train_set(train_data1,train_labels1_iter0,5001,"accuracy1_0.70_iter0.txt","Model1/")
train_set(train_data2,train_labels2_iter0,5001,"accuracy2_0.70_iter0.txt","Model2/")

train_labels1_iter1_model1 = update_labels(train_data1,train_labels1_iter0,"Model1")
train_labels1_iter1_model2 = update_labels(train_data1,train_labels1_iter0,"Model2")
train_labels2_iter1_model1 = update_labels(train_data2,train_labels2_iter0,"Model1")
train_labels2_iter1_model2 = update_labels(train_data2,train_labels2_iter0,"Model2")

train_labels1_iter1 = update_cross_shuffle(train_labels1_iter1_model1,train_labels1_iter1_model2)
train_labels2_iter1 = update_cross_shuffle(train_labels2_iter1_model2,train_labels2_iter1_model1)

labels_dict = {}
labels_dict["train_labels1_iter1"] = train_labels1_iter1
labels_dict["train_labels2_iter1"] = train_labels2_iter1
#print labels_dict["train_labels1_iter1"].shape
#print labels_dict["train_labels1_iter{0}".format(1)].shape

for iter_index in range(1,10):
  print "iter_index:", iter_index
  train_set(train_data1,labels_dict["train_labels1_iter{0}".format(iter_index)], 10001, "accuracy1_0.70_iter{0}.txt".format(iter_index), "Model1/")
  train_set(train_data2,labels_dict["train_labels2_iter{0}".format(iter_index)], 10001, "accuracy2_0.70_iter{0}.txt".format(iter_index), "Model2/")
  
  labels_dict["train_labels1_iter{0}_model1".format(iter_index+1)] = update_labels(train_data1,labels_dict["train_labels1_iter{0}".format(iter_index)],"Model1")
  labels_dict["train_labels1_iter{0}_model2".format(iter_index+1)] = update_labels(train_data1,labels_dict["train_labels1_iter{0}".format(iter_index)],"Model2")
  labels_dict["train_labels2_iter{0}_model1".format(iter_index+1)] = update_labels(train_data2,labels_dict["train_labels2_iter{0}".format(iter_index)],"Model1")
  labels_dict["train_labels2_iter{0}_model2".format(iter_index+1)] = update_labels(train_data2,labels_dict["train_labels2_iter{0}".format(iter_index)],"Model2")

  labels_dict["train_labels1_iter{0}".format(iter_index+1)] = update_cross_shuffle(labels_dict["train_labels1_iter{0}_model1".format(iter_index+1)],labels_dict["train_labels1_iter{0}_model2".format(iter_index+1)])
  labels_dict["train_labels2_iter{0}".format(iter_index+1)] = update_cross_shuffle(labels_dict["train_labels2_iter{0}_model2".format(iter_index+1)],labels_dict["train_labels2_iter{0}_model1".format(iter_index+1)])

'''
for x in range(1,10):
  train_set(train_data1,globals()['train_labels1_iter%s' x],11,['accuracy1_0.70_iter%s.txt' x],['Model1_iter%s/' x])
  train_set(train_data2,globals()['train_labels2_iter%s' x],11,['accuracy2_0.70_iter%s.txt' x],['Model2_iter%s/' x])
  
  globals()['train_labels1_iter%s_model1' x+1] = update_labels(train_data1,globals()['train_labels1_iter%s' x],['Model1_iter%s/' x])
  globals()['train_labels1_iter%s_model2' x+1] = update_labels(train_data1,globals()['train_labels1_iter%s' x],['Model1_iter%s/' x])
  globals()['train_labels2_iter%s_model1' x+1] = update_labels(train_data2,globals()['train_labels2_iter%s' x],['Model1_iter%s/' x])
  globals()['train_labels2_iter%s_model2' x+1] = update_labels(train_data2,globals()['train_labels2_iter%s' x],['Model1_iter%s/' x])

  globals()['train_labels1_iter%s' x+1] = update_cross_shuffle(globals()['train_labels1_iter%s_model1' x+1],globals()['train_labels1_iter%s_model2' x+1])
  globals()['train_labels2_iter%s' x+1] = update_cross_shuffle(globals()['train_labels2_iter%s_model2' x+1],globals()['train_labels2_iter%s_model1' x+1])
'''
'''
# iter1
train_set(train_data1,train_labels1_iter1,10001,"accuracy1_0.70_iter1.txt","Model1_iter1/")
train_set(train_data2,train_labels2_iter1,10001,"accuracy2_0.70_iter1.txt","Model2_iter1/")

train_labels1_iter2_model1 = update_labels(train_data1,train_labels1_iter1,"Model1_iter1")
train_labels1_iter2_model2 = update_labels(train_data1,train_labels1_iter1,"Model2_iter1")
train_labels2_iter2_model1 = update_labels(train_data2,train_labels2_iter1,"Model1_iter1")
train_labels2_iter2_model2 = update_labels(train_data2,train_labels2_iter1,"Model2_iter1")

train_labels1_iter2 = update_cross_shuffle(train_labels1_iter2_model1,train_labels1_iter2_model2)
train_labels2_iter2 = update_cross_shuffle(train_labels2_iter2_model2,train_labels2_iter2_model1)

# iter2
train_set(train_data1,train_labels1_iter2,10001,"accuracy1_0.70_iter2.txt","Model1_iter2/")
train_set(train_data2,train_labels2_iter2,10001,"accuracy2_0.70_iter2.txt","Model2_iter2/")

train_labels1_iter3_model1 = update_labels(train_data1,train_labels1_iter2,"Model1_iter2")
train_labels1_iter3_model2 = update_labels(train_data1,train_labels1_iter2,"Model2_iter2")
train_labels2_iter3_model1 = update_labels(train_data2,train_labels2_iter2,"Model1_iter2")
train_labels2_iter3_model2 = update_labels(train_data2,train_labels2_iter2,"Model2_iter2")

train_labels1_iter3 = update_cross_shuffle(train_labels1_iter3_model1,train_labels1_iter3_model2)
train_labels2_iter3 = update_cross_shuffle(train_labels2_iter3_model2,train_labels2_iter3_model1)

# iter3
train_set(train_data1,train_labels1_iter3,10001,"accuracy1_0.70_iter3.txt","Model1_iter3/")
train_set(train_data2,train_labels2_iter3,10001,"accuracy2_0.70_iter3.txt","Model2_iter3/")

train_labels1_iter4_model1 = update_labels(train_data1,train_labels1_iter3,"Model1_iter3")
train_labels1_iter4_model2 = update_labels(train_data1,train_labels1_iter3,"Model2_iter3")
train_labels2_iter4_model1 = update_labels(train_data2,train_labels2_iter3,"Model1_iter3")
train_labels2_iter4_model2 = update_labels(train_data2,train_labels2_iter3,"Model2_iter3")

train_labels1_iter4 = update_cross_shuffle(train_labels1_iter4_model1,train_labels1_iter4_model2)
train_labels2_iter4 = update_cross_shuffle(train_labels2_iter4_model2,train_labels2_iter4_model1)
'''