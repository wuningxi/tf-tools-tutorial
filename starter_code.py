# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier as starter code for Tensorboard tutorial."""

import argparse
import os
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


# Create a multilayer model.

def train():
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, fake_data=FLAGS.fake_data)

    # Define TF graph

    # Input placeholders
    x = tf.placeholder(tf.float32, [None, 784], name='x-input') # flattened images, original dimensions 28 x 28
    y_ = tf.placeholder(tf.int64, [None], name='y-input')

    # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def nn_layer(input_tensor, input_dim, output_dim, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        # This Variable will hold the state of the weights for the layer
        weights = weight_variable([input_dim, output_dim])
        biases = bias_variable([output_dim])
        preactivate = tf.matmul(input_tensor, weights) + biases
        activations = act(preactivate, name='activation')
        return activations

    hidden1 = nn_layer(x, 784, 500)

    keep_prob = tf.placeholder(tf.float32)
    dropped = tf.nn.dropout(hidden1, keep_prob)

    # Do not apply softmax activation yet, see below.
    y = nn_layer(dropped, 500, 10, act=tf.identity)

    # The raw formulation of cross-entropy,
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)), reduction_indices=[1]))
    # can be numerically unstable.
    # So here we use tf.losses.sparse_softmax_cross_entropy on the
    # raw logit outputs of the nn_layer above, and then average across
    # the batch.
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)

    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Execute graph
    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        # Train the model, and also write summaries.
        # Every 10th step, measure test-set accuracy, and write test summaries
        # All other steps, run train_step on training data, & add training summaries

        def feed_dict(train):
            """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
            if train or FLAGS.fake_data:
                xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
                k = FLAGS.dropout
            else:
                xs, ys = mnist.test.images, mnist.test.labels
                k = 1.0
            return {x: xs, y_: ys, keep_prob: k}

        for i in range(FLAGS.max_steps):
            if i % 10 == 0:  # Record summaries and test-set accuracy
                acc = sess.run([accuracy], feed_dict=feed_dict(False))
                print('Accuracy at step %s: %s' % (i, acc))
            else:  # Record train set summaries, and train
                 _ = sess.run([train_step], feed_dict=feed_dict(True))

def main(_):
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/mnist/input_data'),
        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
