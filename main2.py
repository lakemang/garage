import tensorflow as tf
from skimage import transform
from skimage import data
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.color import rgb2gray
import random

# Import `tensorflow`
import tensorflow as tf


def load_data(data_dir):
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".jpg")]
        for f in file_names:
            images.append(data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "/home/hguan/project/machine-learning/garage/"
train_data_dir = os.path.join(ROOT_PATH, "training")
test_data_dir = os.path.join(ROOT_PATH, "testing")

images, labels = load_data(train_data_dir)

images_array = np.array(images)
labels_array = np.array(labels)
print(len(set(labels_array)))

# Resize images
images32 = [transform.resize(image, (28, 28)) for image in images]
images32 = np.array(images32)
images32 = rgb2gray(np.array(images32))
print(images32.shape)

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 2
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 2]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_classes', 2, 'Number of classes.')
flags.DEFINE_integer('num_variables', 784, 'Number of variables.')

# Hyper Parameters
flags.DEFINE_integer('hidden1', 2048, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 1024, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('num_epochs', 200, 'Number of learning epochs.')
flags.DEFINE_integer('batch_size', 90, 'Batch size.')
flags.DEFINE_float('keep_prob', 0.5, 'Keep probability for drop out.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')

def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

def training(loss):
    tf.summary.scalar(loss.op.name, loss)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def run_training(data, labels):
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        data_size = FLAGS.num_variables
        num_classes = FLAGS.num_classes
        w = tf.Variable(tf.zeros([784, 2])
        b = tf.Variable(tf.zeros([2]))

        # Build a Graph that computes predictions from the inference model.
        logits = tf.matmul(data, w) + b

        # Add to the Graph the loss calculation.
        loss_op = loss(logits, labels_placeholder)

        # Add to the Graph operations that train the model.
        train_op = training(loss_op)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = evaluation(logits, labels_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # check point
        saver = tf.train.Saver()

        # Create a session for running operations in the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter('.', sess.graph)

        # Initialize the variables (the trained variables and the epoch counter).
        sess.run(init_op)

        # training
        for epoch in range(FLAGS.num_epochs):
            for i in range(int(len(data)/FLAGS.batch_size)):
                batch = FLAGS.batch_size*i
                sess.run(train_op, feed_dict={
                    data_placeholder: data[batch:batch + FLAGS.batch_size],
                    labels_placeholder: labels[batch:batch + FLAGS.batch_size],
                    keep_prob: FLAGS.keep_prob})

            # calculate accuracy in every epoch
            train_accuracy = sess.run(eval_correct, feed_dict={
                data_placeholder: data,
                labels_placeholder: labels,
                keep_prob: FLAGS.keep_prob})
            print("epoch %d, acc %g" % (epoch, train_accuracy / len(labels)))

            # update TensorBoard
            summary_str = sess.run(summary, feed_dict={
                data_placeholder: data,
                labels_placeholder: labels,
                keep_prob: 1.0})
            summary_writer.add_summary(summary_str, epoch)

            # shuffling data for next epoch
            data, labels = shuffle_data(data, labels)

        # Save a checkpoint and evaluate the model periodically.
        # Create a saver for writing training checkpoints.
        path = saver.save(sess, 'models.ckpt')
        print('checkpoint is saved at ' + path)

        sess.close()

# Start training
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        # Fit training using batch data
        _, avg_cost = sess.run([optimizer, cost], feed_dict={x: images32,
                                                      y: labels})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#    print "Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]})

