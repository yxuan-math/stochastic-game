import logging
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import eval
TF_DTYPE = tf.float64
DELTA_CLIP = 100.0
MOMENTUM = 0.99
EPSILON = 1e-6

class CovidMultiModel(object):
    # all the networks share the same parameters and take t as an additional parameter
    # train multiple networks
    def __init__(self, config, bsde, sess):
        self.eqn_config = config.eqn_config
        self.nn_config = config.nn_config
        self.y_hiddens = self.nn_config.num_hiddens.copy()
        self.y_hiddens.append(1)
        self.bsde = bsde
        self.sess = sess
        # make sure consistent with FBSDE equation
        self.n_player = bsde.n_player
        self.num_time_interval = bsde.num_time_interval
        self.total_time = bsde.total_time
        self.log_msg = ''
        self.train_hist_header = ''
        self.x_fake = np.zeros(
            [self.nn_config.batch_size, config.eqn_config.dim_x, self.num_time_interval + 1])
        self.y_init = [None for _ in range(self.n_player)]
        self.y_list = [[] for _ in range(self.n_player)]
        self.z_list = [[] for _ in range(self.n_player)]
        self.l_list= [[] for _ in range(self.n_player)]
        self.train_ops = [[] for _ in range(self.n_player)]
        self.relerr_y, self.relerr_z = [0] * self.n_player, [0] * self.n_player
        self.player_loss = [None] * self.n_player
        self.fetch_list = [None] * self.n_player
        self.policy_func = None
        self.dw = tf.placeholder(TF_DTYPE,
                                 [None, config.eqn_config.dim_w, self.num_time_interval], name='dW')
        self.x = tf.placeholder(TF_DTYPE,
                                [None, config.eqn_config.dim_x, self.num_time_interval+1], name='X')
        self.old_policy_evaluate= tf.placeholder(TF_DTYPE,
                                [None, self.n_player, self.num_time_interval], name='policy')
        self.is_training = tf.placeholder(tf.bool)


    def train_play(self, policy_func):
        # to save iteration results
        training_history = []
        player = 0
        valid_player = 0
        policy_freq = self.nn_config.policy_freq
        dw_train, x_train, old_policy_train = [None] * policy_freq*self.n_player, [None] * policy_freq*self.n_player,[None] * policy_freq*self.n_player
        # initialization
        self.sess.run(tf.global_variables_initializer())
        # begin sgd iteration
        for step in range(self.nn_config.num_iterations*self.n_player+1):
            if step % (policy_freq*self.n_player) == 0:
                for i in range(self.n_player):
                    for j in range(policy_freq):
                        dw_train[i*policy_freq+j], x_train[i*policy_freq+j],old_policy_train[i*policy_freq+j] = self.bsde.sample(
                            self.nn_config.batch_size, policy_func, i)
                #tmp_p = np.random.randint(self.n_player)
            idx = step % (policy_freq*self.n_player)
            player = idx // policy_freq

            self.sess.run(self.train_ops[player],
                        feed_dict={self.dw: dw_train[idx], self.x: x_train[idx],self.old_policy_evaluate:old_policy_train[idx],
                                    self.is_training: True})

            if (step+1) % self.nn_config.logging_frequency == 0:

                valid_player =player
                dw_valid, x_valid,old_policy_valid = self.bsde.sample(
                    self.nn_config.valid_size, policy_func, valid_player)

                fetch_data = self.sess.run(self.fetch_list[valid_player], feed_dict={self.dw: dw_valid, self.x: x_valid,self.old_policy_evaluate:old_policy_valid,self.is_training: False})
                if self.nn_config.verbose:
                    print((step, valid_player) + tuple(fetch_data))
                    logging.info(("step: %5u    pl: %1d     " + self.log_msg) % ((step, valid_player) + tuple(fetch_data)))
                    training_history.append((step, valid_player) + tuple(fetch_data))
        return np.array(training_history)



    def build(self):
        with tf.variable_scope('forward'):
            for i in range(self.n_player):
                print('###########player',i)
                with tf.variable_scope('player%d' % i):
                    x_init = self.x[:, :, 0]

                    self.y_init[i], z = self._subnetwork_grad(x_init, 0, self.y_hiddens, name='player%d' % i)

                    l=self._subnetwork_policy(x_init, 0, self.y_hiddens, name='policy_player%d' % i)
                    y = self.y_init[i]
                    self.y_list[i].append(y)
                    self.z_list[i].append(z)
                    self.l_list[i].append(l)
                    # z_true = self.bsde.true_z(x_init, 0, player=i)
                    # z_diff = tf.reduce_mean((z - z_true) ** 2)
                    #z_mean = tf.reduce_mean(z_true)
                    #z2_mean = tf.reduce_mean(z_true ** 2)
                    delta_l=0
                    for t in range(0, self.num_time_interval - 1):
                        l_star,h=self.new_policy(self.x[:, :, t],self.old_policy_evaluate[:,:,t],z,t,player=i)
                        delta_l+=tf.square(l-l_star)
                        y = y - self.bsde.delta_t * h + \
                            tf.reduce_sum(z * self.dw[:, :, t], 1, keepdims=True)
                        _, z = self._subnetwork_grad(
                            self.x[:, :, t + 1], (t + 1)/self.num_time_interval*self.total_time,
                            self.y_hiddens, name='player%d' % i)
                        l = self._subnetwork_policy(self.x[:, :, t + 1], (t + 1)/self.num_time_interval*self.total_time, self.y_hiddens, name='policy_player%d' % i)
                        self.z_list[i].append(z)
                        self.y_list[i].append(y)
                        self.l_list[i].append(l)

                    # terminal time
                    l_star,h=self.new_policy(self.x[:, :, -2],self.old_policy_evaluate[:,:,-1],z,self.num_time_interval - 1,player=i)
                    delta_l+=tf.square(l-l_star)
                    y = y - self.bsde.delta_t * h+ \
                        tf.reduce_sum(z * self.dw[:, :, -1], 1, keepdims=True)

                    y_xT=0
                    delta = tf.square(y - y_xT)+self.nn_config.weight_match*delta_l/self.num_time_interval

                    self.player_loss[i] = tf.reduce_mean(delta)
                    # train operations
                    global_step = tf.get_variable('global_step', [],
                                                  initializer=tf.constant_initializer(0),
                                                  trainable=False, dtype=tf.int32)
                    learning_rate = tf.train.piecewise_constant(global_step,
                                                                self.nn_config.lr_boundaries,
                                                                self.nn_config.lr_values)
                    # we use constant learning rate; if want to use decreasing learning rate:  
                    # self.train_ops[i] = tf.train.AdamOptimizer(learning_rate).minimize(self.player_loss[i],global_step=global_step)                                          
                    self.train_ops[i] = tf.train.AdamOptimizer(learning_rate).minimize(self.player_loss[i])
                    
            for i in range(self.n_player):

                self.fetch_list[i] = [self.player_loss[i]]
            self.log_msg = 'valid_loss: %.4e'
            self.train_hist_header = 'step, valid_player, valid_loss'
            self.old_policy_func=self.old_policy
            
    def build_restore(self):

        with tf.variable_scope('aforward'):
            for i in range(self.n_player):
                print('###########player',i)
                with tf.variable_scope('player%d' % i):
                    x_init = self.x[:, :, 0]

                    self.y_init[i], z = self._subnetwork_grad(x_init, 0, self.y_hiddens, name='player%d' % i)

                    l=self._subnetwork_policy(x_init, 0, self.y_hiddens, name='policy_player%d' % i)
                    y = self.y_init[i]
                    self.y_list[i].append(y)
                    self.z_list[i].append(z)
                    self.l_list[i].append(l)
                    delta_l=0
                    for t in range(0, self.num_time_interval - 1):
                        l_star,h=self.new_policy(self.x[:, :, t],self.old_policy_evaluate[:,:,t],z,t,player=i)
                        delta_l+=tf.square(l-l_star)
                        y = y - self.bsde.delta_t * h + \
                            tf.reduce_sum(z * self.dw[:, :, t], 1, keepdims=True)
                        _, z = self._subnetwork_grad(
                            self.x[:, :, t + 1], (t + 1)/self.num_time_interval*self.total_time,
                            self.y_hiddens, name='player%d' % i)
                        l = self._subnetwork_policy(self.x[:, :, t + 1], (t + 1)/self.num_time_interval*self.total_time, self.y_hiddens, name='policy_player%d' % i)
                        self.z_list[i].append(z)
                        self.y_list[i].append(y)
                        self.l_list[i].append(l)

                    l_star,h=self.new_policy(self.x[:, :, -2],self.old_policy_evaluate[:,:,-1],z,self.num_time_interval - 1,player=i)
                    delta_l+=tf.square(l-l_star)
                    y = y - self.bsde.delta_t * h+ \
                        tf.reduce_sum(z * self.dw[:, :, -1], 1, keepdims=True)

                    y_xT=0
                    delta = tf.square(y - y_xT)+self.nn_config.weight_match*delta_l/self.num_time_interval

                    self.player_loss[i] = tf.reduce_mean(delta)

            for i in range(self.n_player):
                self.fetch_list[i] = [self.player_loss[i]]
            self.log_msg = 'valid_loss: %.4e'
            self.train_hist_header = 'step, valid_player, valid_loss'
            self.old_policy_func=self.old_policy

    def old_policy(self, x, t_ind):
        old_policy=np.zeros([self.nn_config.batch_size,self.n_player])
        self.x_fake[:, :, t_ind] = x
        for idx in range(0, self.n_player):
             l = self.sess.run(self.l_list[idx][t_ind],
                                  feed_dict={self.x: self.x_fake, self.is_training: False})
             old_policy[:,idx]=l[:,0]
        return old_policy

    def new_policy(self, x, T, z, t,player):
        # return policy evaluation for all the players except player

        s_m=tf.expand_dims(x[:,player], axis=1)
        e_m=tf.expand_dims(x[:,self.n_player+player], axis=1)
        i_m=tf.expand_dims(x[:,2*self.n_player+player], axis=1)
        v_sm=tf.expand_dims(z[:,player], axis=1)
        v_em=tf.expand_dims(z[:,self.n_player+player], axis=1)
        A=self.bsde.theta**2*self.bsde.beta[player,player]*s_m*i_m*(v_em-v_sm)
        B_1=tf.reduce_sum(self.bsde.theta*(1-self.bsde.theta*T)*self.bsde.beta[player,:]*x[:,(-1*self.n_player):], 1, keepdims=True)*s_m*(v_em-v_sm)
        B_2=tf.reduce_sum(self.bsde.theta*(1-self.bsde.theta*T)*self.bsde.beta[:,player]*x[:,:self.n_player]*(z[:,self.n_player:(2*self.n_player)]-z[:,:self.n_player]), 1, keepdims=True)*i_m
        B=np.exp(-1*self.bsde.r*(t)/self.num_time_interval*self.total_time)*self.bsde.population[player]*(s_m+e_m+i_m)*self.bsde.w-B_1-B_2
        helper_0=np.zeros((self.nn_config.batch_size,1))
        helper_1=np.ones((self.nn_config.batch_size,1))
        l_star=tf.where(A<=0,tf.where(A+B<=0,helper_1,helper_0),tf.maximum(np.float64(0),tf.minimum(np.float64(1),-1*B/(2*A))))
        c=np.exp(-1*self.bsde.r*(t)/self.num_time_interval*self.total_time)*self.bsde.population[player]*(self.bsde.k*self.bsde.chi+self.bsde.h*self.bsde.q)*i_m
        h=A*(l_star*l_star)+B*l_star+c
        return l_star,h
    

    def _dense_batch_layer(self, input_, output_size, activation_fn=None,
                     stddev=5.0, name='linear'):
        with tf.variable_scope(name):
            hiddens = tf.layers.dense(input_, output_size, activation=None)
            hiddens = tf.layers.batch_normalization(hiddens)
            if activation_fn:
                return activation_fn(hiddens)
            else:
                return hiddens

    def _subnetwork_grad(self, x, t, num_hiddens, name='subnet_grad'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # standardize the path input first
            # the affine  could be redundant, but helps converge faster
            hiddens = tf.layers.batch_normalization(x)
            #hiddens=x
            t = tf.constant(np.broadcast_to(t, [self.nn_config.batch_size, 1]), dtype=TF_DTYPE)
            hiddens = tf.concat([hiddens, t], axis=1)
            for i in range(1, len(num_hiddens) - 1):
                hiddens = self._dense_batch_layer(hiddens,
                                                  num_hiddens[i],
                                                  activation_fn=tf.nn.tanh,
                                                  name='layer_{}'.format(i))
            y_out = self._dense_batch_layer(hiddens,
                                            num_hiddens[-1],
                                            activation_fn=None,
                                            name='final_layer')
            z_out = tf.gradients(y_out, x)[0]

        return y_out, z_out
    
    def _subnetwork_policy(self, x, t, num_hiddens, name='subnet_policy'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # standardize the path input first
            # the affine  could be redundant, but helps converge faster
            hiddens = tf.layers.batch_normalization(x)
            #hiddens=x
            t = tf.constant(np.broadcast_to(t, [self.nn_config.batch_size, 1]), dtype=TF_DTYPE)
            hiddens = tf.concat([hiddens, t], axis=1)
            for i in range(1, len(num_hiddens) - 1):
                hiddens = self._dense_batch_layer(hiddens,
                                                    num_hiddens[i],
                                                    activation_fn=tf.nn.tanh,
                                                    name='layer_{}'.format(i))
            y_out = self._dense_batch_layer(hiddens,
                                            num_hiddens[-1],
                                            activation_fn=tf.nn.sigmoid,
                                            name='final_layer')

        return y_out

