import pickle
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet

# Load traffic signs data.
with open('/home/carnd/self_driving_car_nd/term1/CarND-Alexnet-Feature-Extraction-Lab/train.p', 'rb') as reader:
    train = pickle.load(reader)
    
# Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(train['features'][:1000,], train['labels'][:1000], test_size = 0.2)

# Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized_x = tf.image.resize_images(x, (227, 227))

y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

# Pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized_x, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], 43)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(43))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(logits)

# Parameters

EPOCHS = 5
BATCH_SIZE = 200

# TODO: Define loss, training, accuracy operations.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer()

training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

# TODO: Train and evaluate the feature extraction model.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    t0 = time.time()
    print("Training starting at {}...".format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))
    
    for i in range(EPOCHS):
        count = 0
        t1 = time.time()
        X_train_model, y_train_model = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            count += 1
            if count%10 == 0:
                print("EPOCH: {}; BATCH: {}; TIME ELAPSED IN EPOCH: {}".format(i,count, time.time()-t1))
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_model[offset:end], y_train_model[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        
        ## compute the accuracy for the training/ validation sets
        validation_accuracy = sess.run(accuracy_operation, feed_dict={x: X_valid, y: y_valid})
        
        print("EPOCH {}; TOTAL TIME: {}".format(i+1, time.time()-t0))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
