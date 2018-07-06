import tensorflow as tf
from skimage import transform
from skimage import data
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.color import rgb2gray
import random
import pandas as pd
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

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_classes', 2, 'Number of classes.')
flags.DEFINE_integer('num_variables', 784, 'Number of variables.')

# Hyper Parameters
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('num_epochs', 50, 'Number of learning epochs.')
flags.DEFINE_integer('batch_size', 10, 'Batch size.')
flags.DEFINE_float('keep_prob', 0.5, 'Keep probability for drop out.')
flags.DEFINE_float('learning_rate', 0.05, 'Initial learning rate.')

def inference(data, data_size, keep_prob):
    # # Hidden layer 1
    # with tf.name_scope('hidden1'):
    #     weights = tf.Variable(tf.truncated_normal([data_size, FLAGS.hidden1],
    #                                               stddev=1.0 / math.sqrt(float(data_size))), name='weights1')
    #     biases = tf.Variable(tf.zeros([FLAGS.hidden1]), name='biases1')
    #     hidden1 = tf.nn.relu(tf.matmul(data, weights) + biases)
    #
    #     # Dropout before layer 2
    #     hidden1_drop = tf.nn.dropout(hidden1, keep_prob, name='layer1_dropout')
    #
    # # Hidden layer 2
    # with tf.name_scope('hidden2'):
    #     weights = tf.Variable(tf.truncated_normal([FLAGS.hidden1, FLAGS.hidden2],
    #                                               stddev=1.0 / math.sqrt(float(FLAGS.hidden1))), name='weights2')
    #     biases = tf.Variable(tf.zeros([FLAGS.hidden2]), name='biases2')
    #     hidden2 = tf.nn.sigmoid(tf.matmul(hidden1_drop, weights) + biases)
    #
    #     # Dropout before linear reading out
    #     hidden2_drop = tf.nn.dropout(hidden2, keep_prob, name='layer2_dropout')
    #
    # # Read out
    # with tf.name_scope('softmax_linear'):
    #     weights = tf.Variable(tf.truncated_normal([FLAGS.hidden2, FLAGS.num_classes],
    #                                               stddev=1.0 / math.sqrt(float(FLAGS.hidden2))), name='weights')
    #     biases = tf.Variable(tf.zeros([FLAGS.num_classes]), name='biases')
    #     logits = tf.matmul(hidden2_drop, weights) + biases

        weights = tf.Variable(tf.zeros([data_size, FLAGS.num_classes]), name='weights')
        biases = tf.Variable(tf.zeros([FLAGS.num_classes]), name='biases')
        logits = tf.matmul(data, weights) + biases

        return logits

def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
    vars = tf.trainable_variables()
    reg = tf.add_n([tf.nn.l2_loss(v) for v in vars])*0.1
#    reg = 0.01*(tf.nn.l2_loss(w) + tf.nn.l2_loss(b))
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean') + reg
    return loss

def training(loss):
    tf.summary.scalar(loss.op.name, loss)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

def shuffle_data(data, labels):
    # transform list into DataFrame
    new_df = pd.DataFrame(data)
    new_df['__labels__'] = labels
    new_df = new_df.reindex(np.random.permutation(new_df.index))

    new_labels = list(new_df['__labels__'])
    del new_df['__labels__']

    # transform DataFrame into list
    new_row = []
    for index, row in new_df.iterrows():
        _list_row = []
        for col in new_df:
            _list_row.append(row[col])
        new_row.append(_list_row)

    return new_row, new_labels

def run_training(data, labels):
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        data_size = FLAGS.num_variables
        num_classes = FLAGS.num_classes

        data_placeholder = tf.placeholder("float", shape=(None, data_size))
        labels_placeholder = tf.placeholder("int32", shape=None)
        keep_prob = tf.placeholder("float")

        # Build a Graph that computes predictions from the inference model.
        logits = inference(data_placeholder, data_size, keep_prob)

        correct_pred = tf.argmax(logits, 1)

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
            actual_loss, train_accuracy = sess.run([loss_op, eval_correct], feed_dict={
                data_placeholder: data,
                labels_placeholder: labels,
                keep_prob: FLAGS.keep_prob})
            print("epoch %d, loss %g, acc %g" % (epoch, actual_loss, train_accuracy / len(labels)))

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
        path = saver.save(sess, '/home/hguan/project/machine-learning/garage/models.ckpt')
        print('checkpoint is saved at ' + path)

        sess.close()

def run_classifier(data):
    with tf.Graph().as_default():
        data_size = FLAGS.num_variables
        num_classes = FLAGS.num_classes
        data_placeholder = tf.placeholder("float", shape=(None, data_size))

        logits = inference(data_placeholder, data_size, 1.0)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess = tf.Session()
        sess.run(init_op)

        saver = tf.train.Saver()
        saver.restore(sess, '/home/hguan/project/machine-learning/garage/models.ckpt')

        prediction_list = []
        total = len(data)
        with sess.as_default():
            for i in range(total):
                print('#%d : ' % i, end='')
                v = logits.eval(feed_dict={data_placeholder: [data[i]]})
                sm = tf.nn.softmax(v)
                smv = sm.eval()
#                prediction_list.append(smv[0])

                cls = np.argmax(smv)   # get the class number with largest value by argmax
                prediction_list.append(cls)
                print('prediction=%d (%f)' % (cls, smv[0][cls]))

        return prediction_list

def test_data():
    # Testing data
    images, labels = load_data(test_data_dir)
    images_array = np.array(images)
    labels_array = np.array(labels)

    # Resize images
    images32 = [transform.resize(image, (28, 28)) for image in images]
    images32 = np.array(images32)
    images32 = rgb2gray(np.array(images32))
    images32 = images32.reshape(-1, 784)
    return images32


ROOT_PATH = "/home/hguan/project/machine-learning/garage/"
train_data_dir = os.path.join(ROOT_PATH, "training")
test_data_dir = os.path.join(ROOT_PATH, "testing")

# Training data
images, labels = load_data(test_data_dir)
images_array = np.array(images)
labels_array = np.array(labels)

# Resize images
images32 = [transform.resize(image, (28, 28)) for image in images]
images32c = np.array(images32)
images32 = rgb2gray(np.array(images32c))
images32 = images32.reshape(-1, 784)

#run_training(images32, labels)

#images32 = test_data()
predicted = run_classifier(images32)

# Print the real and predicted labels
print(labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(16, 16))
for i in range(len(images)):
#    print('i=', i)
    truth = labels[i]
    prediction = predicted[i]
    plt.subplot(20, 6,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(700, 300, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(images[i])

plt.show()
