import tensorflow as tf
import numpy as np
from pprint import pprint
import math



# Create TensorFlow object called hello_constant
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)

## place holders and using feed_dict
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run([x,y], feed_dict={x: 'Test String', y: 123, z: 45.67})

## initialize the state of all variable tensors
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

## tf.truncated_normal() to generate numbers from a normal dist
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))

n_labels = 5
bias = tf.Variable(tf.zeros(n_labels))

## softmax function
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # TODO: Compute and return softmax(x)
    if isinstance(x, list):
        return np.exp(x)/np.sum(np.exp(x))
    elif isinstance(x, np.ndarray):
        return np.exp(x)/np.sum(np.exp(x), axis = 0)

## tensorflow softmax
tf.nn.softmax(x)
    
## cross entropy using tensorflow
softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# TODO: Print cross entropy from session
with tf.Session() as sess:
    output = sess.run(-tf.reduce_sum(tf.multiply(one_hot,tf.log(softmax))), 
    feed_dict={
        softmax: softmax_data,
        one_hot:one_hot_data
        })
    print(output)

## in order to do mini-batching for SGD
# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])
## in this case the None serves as a place holder that can be replaced by anything greater than 0 later on

# 4 Samples of features
example_features = [
    ['F11','F12','F13','F14'],
    ['F21','F22','F23','F24'],
    ['F31','F32','F33','F34'],
    ['F41','F42','F43','F44']]
# 4 Samples of labels
example_labels = [
    ['L11','L12'],
    ['L21','L22'],
    ['L31','L32'],
    ['L41','L42']]

def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    results = []
    counter = 0 
    while counter < len(features):
        if (counter+batch_size) <= len(features):
            results.append([features[counter:counter+batch_size], labels[counter:counter+batch_size]])
        else:
            results.append([features[counter:], labels[counter:]])
        counter += batch_size
    return results
