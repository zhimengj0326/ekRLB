
import numpy as np
import tensorflow as tf


class MLP_model:
    """
    This clas creates and manages the networks which will be incorporated
    into the agent. The structure of the network(s) follow Wu et al. (2018),
    with three hidden layers with 100 neurons each.
    """
    def __init__(self, state_size, action_size, learning_rate, hidden_num, hidden_unit, norm_state):
        """
        :param state_size: the dimensionality of the state, which determines
        the size of the input
        :param action_size: the number of possible actions, which determines
        the size of the output
        :param variable_scope: categorizes the names of the tf-variables for
        the local network and the target network.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.norm_state = norm_state

        self.input_pl = tf.compat.v1.placeholder(dtype=np.float32, shape=(None, self.state_size),
                                       name= 'input_pl')
        self.target_pl = tf.compat.v1.placeholder(dtype=np.float32, shape=(None, self.action_size),
                                        name='output_pl')
        self.hidden_layer = self.input_pl / self.norm_state
        # Batch normalization
        for i in range(hidden_num):
            self.hidden_layer = tf.compat.v1.layers.dense(self.hidden_layer, hidden_unit,
                                        activation=tf.nn.tanh, kernel_initializer=tf.compat.v1.initializers.random_normal,
                                        bias_initializer=tf.compat.v1.initializers.random_normal)

        #self.norm_input_pl = tf.compat.v1.layers.batch_normalization(self.norm_input_pl, training=True)
        # self.first_hidden_layer = tf.keras.layers.Dense(100, activation=tf.nn.relu)(self.norm_input_pl)
        # # Batch normalization
        # #self.first_hidden_layer = tf.compat.v1.layers.batch_normalization(self.first_hidden_layer, training=True)
        #
        # self.second_hidden_layer = tf.keras.layers.Dense(100, activation=tf.nn.relu)(self.first_hidden_layer)

        # self.output_layer = tf.keras.layers.Dense(self.second_hidden_layer, action_size,
        #                                 activation=tf.nn.relu, kernel_initializer=tf.compat.v1.initializers.random_normal,
        #                                 bias_initializer=tf.compat.v1.initializers.random_normal,
        #                                 name='.output_layer')
        self.output_layer = tf.compat.v1.layers.dense(self.hidden_layer, action_size,
                                        activation=tf.nn.tanh, kernel_initializer=tf.compat.v1.initializers.random_normal,
                                        bias_initializer=tf.compat.v1.initializers.random_normal,
                                        name='output_layer')
        #tf.summary.scalar('output_layer', tf.reduce_mean(self.output_layer))

        logstd = tf.compat.v1.get_variable(name="logstd", shape=[1, action_size],
                                 initializer=tf.compat.v1.initializers.random_normal)
        self.sample = self.output_layer + logstd * tf.compat.v1.random_normal(tf.shape(self.output_layer))
        #tf.summary.scalar('sample_layer', tf.reduce_mean(self.sample))


        self.loss = tf.losses.mean_squared_error(self.target_pl, self.output_layer)
        #tf.summary.scalar('loss', tf.reduce_mean(self.loss))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate, beta1=0.95).minimize(self.loss)
        self.var_init = tf.compat.v1.global_variables_initializer()
        self.merged = tf.compat.v1.summary.merge_all()

    def predict_single(self, sess, state):
        """
        :param sess: current tf-session used
        :param state: current state for which we want to estimate the value
        of taking certain actions
        :return: estimated value of taking certain actions
        """
        return sess.run(self.output_layer,
                        feed_dict={self.input_pl: np.expand_dims(state, axis=0)})[0]

    def predict_batch(self, sess, states):
        """
        :param sess: current tf-session used
        :param states: batch of states for which we want to estimate values of
        taking certain actions
        :return: estimated values of taking certain actions in a single tensor
        """
        return sess.run(self.output_layer, feed_dict={self.input_pl: states})

    def train_batch(self, sess, inputs, targets):
        """
        :param sess: current tf-session used
        :param inputs: batch of inputs, i.e. states, for which we want to train our
        network
        :param targets: target values with which we want to train our network,
        i.e. estimated returns from taking certain actions
        :return: updated (trained) network
        """
        sess.run(self.optimizer,
                              feed_dict={self.input_pl: inputs, self.target_pl: targets})
