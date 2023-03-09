from __future__ import division

import os
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops


#############################################################################
# Random Orthogonal weight matrix generator #
def gen_ortho_matrix(dim, rng=None):
    '''Generate random orthogonal matrix
    Taken from scipy.stats.ortho_group
    Copied here from compatibilty with older versions of scipy
    '''
    H = np.eye(dim)
    for n in range(1, dim):
        if rng is None:
            x = np.random.normal(size=(dim-n+1,))
        else:
            x = rng.normal(size=(dim-n+1,))
        # random sign, 50/50, but chosen carefully to avoid roundoff error
        D = np.sign(x[0])
        x[0] += D*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = -D*(np.eye(dim-n+1)
                 - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
    return H

######################## RNN cell #################################

class customRNNCell(object):
    def __init__(self, config):

#        super(customRNNCell, self).__init__(_reuse=reuse)

        self._rnn_type = config['rnn_type']
        self._reuse = False

        if config['rng'] is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = config['rng']

        n_input = config['num_input']
        n_rnn = config['num_rnn']

        if self._rnn_type == 'LeakyRNN':
            self._w_rec_init = config['w_rec_init']

            if config['activation'] == 'softplus':
                self._activation = tf.nn.softplus
                self._bias_start = 0.0
                self._w_in_start = 1.0
                if self._w_rec_init == 'diag':
                    self._w_rec_start= 0.54
                elif self._w_rec_init == 'randortho':
                    self._w_rec_start= 1.0
                elif self._w_rec_init == 'randgauss':
                    self._w_rec_start = 1.0
            elif config['activation'] == 'tanh':
                self._activation = tf.tanh
                self._bias_start = 0.
                self._w_in_start = 1.0
                if self._w_rec_init == 'diag':
                    self._w_rec_start= 0.54
                elif self._w_rec_init == 'randortho':
                    self._w_rec_start= 1.0
                elif self._w_rec_init == 'randgauss':
                    self._w_rec_start = 1.0
            elif config['activation'] == 'relu':
                self._activation = tf.nn.relu
                self._bias_start = 0.5
                self._w_in_start = 1.0
                if self._w_rec_init == 'diag':
                    self._w_rec_start= 0.54
                elif self._w_rec_init == 'randortho':
                    self._w_rec_start= 1.0 #0.5
                elif self._w_rec_init == 'randgauss':
                    self._w_rec_start = 1.0
            elif config['activation'] == 'suplin':
                self._activation = lambda x: tf.square(tf.nn.relu(x))
                self._bias_start = 0.5
                self._w_in_start = 1.0
                if self._w_rec_init == 'diag':
                    self._w_rec_start= 0.01 # Only using this now
                elif self._w_rec_init == 'randortho':
                    self._w_rec_start= 1.0
                elif self._w_rec_init == 'randgauss':
                    self._w_rec_start = 1.0
            else:
                raise ValueError('Unknown activation')

            self._alpha = config['alpha']
            self._alpha_noise = config['alpha_noise']
            self._sigma = config['sigma_rec']

            ################ Building model parameters ####################
            w_in2rnn0 = self.rng.randn(n_input, n_rnn) / np.sqrt(n_input) * self._w_in_start

            if self._w_rec_init == 'diag':
                w_rec0 = self._w_rec_start*np.eye(n_rnn) # VG - No normalization by sqrt(n_hidden) here??
            elif self._w_rec_init == 'randortho':
                w_rec0 = self._w_rec_start*gen_ortho_matrix(n_rnn, rng=self.rng) # VG - No normalization by sqrt(n_hidden) here??
            elif self._w_rec_init == 'randgauss':
                w_rec0 = self._w_rec_start*self.rng.randn(n_rnn, n_rnn)/np.sqrt(n_rnn)

            self.in2rnn = tf.compat.v1.get_variable('RNNin_weights', dtype=tf.float32, shape=[n_input, n_rnn],
                                           initializer=tf.constant_initializer(w_in2rnn0))

            self._kernel = tf.compat.v1.get_variable('leakyRNN_weights', dtype=tf.float32, shape=[n_rnn, n_rnn],
                                           initializer=tf.constant_initializer(w_rec0))
            self._bias = tf.compat.v1.get_variable('leakyRNN_biases', dtype=tf.float32,
                                         shape=[n_rnn],
                                         initializer=init_ops.constant_initializer(self._bias_start))

            self.init_state = tf.compat.v1.get_variable('leakyRNN_init_state', shape=[1, n_rnn], dtype=tf.float32,
                                                   initializer=init_ops.constant_initializer(0.0))
            self.initS = tf.tile(self.init_state, [config['batch_size'], 1])

            ###############################################################

        else:
            raise NotImplementedError()

        # Helper tensors
        self.initNoise = tf.zeros(shape=[config['batch_size'], n_rnn])
        self.zeroStims = tf.zeros(shape=[config['batch_size'], config['num_input']])
        self.onlyFixStim = tf.concat([tf.zeros(shape=[config['batch_size'], np.prod(config['image_shape'])]),
                                      np.float32(config['fixationInput'])*tf.ones(shape=[config['batch_size'], config['num_input'] - np.prod(config['image_shape'])])], axis = 1)

        # Save config parameters
        self.config = config

    def call(self, stims, states0, OUnoise):
        states = states0

        # Single-timestep update of state
        with tf.compat.v1.variable_scope("compute", reuse=self._reuse):
            incs = tf.matmul(stims, self.in2rnn)
            new_noise = (1.0-self._alpha_noise)*OUnoise + tf.random.normal(tf.shape(states), mean=0, stddev=np.sqrt(2.0*self._alpha_noise)*self._sigma, dtype=tf.float32)
            new_states = (1.0 - self._alpha) * states + \
                         self._alpha * self._activation(incs + tf.matmul(states, self._kernel) + self._bias + new_noise)

            ret_states = new_states
            ret_noise = new_noise
        self._reuse = True
        return ret_noise, ret_states


    def unroll(self, imageStims, config):
        # Reinitialize state
        states = self.initS
        noise = self.initNoise

        # Buffer to dump aux variables (inputs, states, etc.) across timesteps within a trial
        in_rnn_ta = tf.TensorArray(dtype=tf.float32,
                                   size=config['tdim'],
                                   tensor_array_name="in_ta")
        output_rnn_ta = tf.TensorArray(dtype=tf.float32,
                                       size=config['tdim'],
                                       tensor_array_name="rnn_ta")

        # Loop through time for forward pass
        for time in range(config['tdim']):
            # Set time-specific inputs
            if time >= config['stimPeriod'][0] and time < config['stimPeriod'][1]:
                stims = imageStims
            elif time < config['fixationPeriod'][1]:
                stims = self.onlyFixStim
            else:
                stims = self.zeroStims

            # Single timestep forward pass
            new_noise, new_states = self.call(stims, states, noise)

            # Dump aux variables
            in_rnn_ta = in_rnn_ta.write(time, stims)
            output_rnn_ta = output_rnn_ta.write(time, new_states)

            states = new_states
            noise = new_noise

        final_rnn_inputs = in_rnn_ta.stack()
        final_rnn_outputs = output_rnn_ta.stack()

        return (final_rnn_outputs, final_rnn_inputs, new_states)


########################################################################

# Network model
class Model(object):
    def __init__(self, config):

        if config['seed'] is not None:
            tf.compat.v1.set_random_seed(config['seed'])
        else:
            print('Warning: Random seed not specified')

        # Setup optimizer learning rates as non-trainable variable so they may be adjusted throughout training, as required
        self.learning_rate_full = tf.Variable(config['init_lr_full'], dtype=tf.float32, trainable=False)

        # Generate placeholders
        self.generateInputs(config)

        # Build graph
        self.buildModel(config)

        # Setup optimization
        self.optimize(config)

        # Setup tf helper functionality
        self.saver = tf.compat.v1.train.Saver()
        self.config = config
        self.sess = None

    # Generate placeholders for inputs, target ouputs and masks
    # Currently assuming cross-entropy loss only
    def generateInputs(self, config):
        # RNN target output and mask
        self.y_rnn = tf.compat.v1.placeholder(tf.float32, [None] + [config['num_rnn_out']])
        self.y_rnn_mask = tf.compat.v1.placeholder(tf.bool,[None])

        # Image inputs
        self.x = tf.compat.v1.placeholder(tf.float32, [config['batch_size'], config['num_input']])

    # Build graph
    def buildModel(self, config):

        # Initialize custom cell object
        cell = customRNNCell(config)

        # Create and initialize output parameters
        with tf.compat.v1.variable_scope("output"):
            self.w_rnn_out = tf.compat.v1.get_variable('out_RNN_weights', shape=[config['num_rnn'], config['num_rnn_out']], dtype=tf.float32,
                                        initializer=init_ops.constant_initializer(0.0))

            self.b_rnn_out = tf.compat.v1.get_variable('out_RNN_biases', shape=[config['num_rnn_out']], dtype=tf.float32,
                                        initializer=init_ops.constant_initializer(0.0))

        # Unroll RNN
        self.final_rnn_outputs, self.final_rnn_inputs, self.final_state = cell.unroll(self.x, config)

        # Generate decision output nodes from RNN state
        self.states = tf.reshape(self.final_rnn_outputs, (-1, config['num_rnn']))
        self.y_hat_ = tf.matmul(self.states, self.w_rnn_out) + self.b_rnn_out
        self.y_hat = tf.nn.softmax(self.y_hat_)

    # Setup optimization
    def optimize(self, config):

        # Update loss
        rnn_err = tf.boolean_mask(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_rnn, logits=self.y_hat_),
                                   self.y_rnn_mask)
        self.cost_lsq_rnn = tf.reduce_mean(rnn_err)

        # Add in regularizers
        # Firing rate / homeostatic regularizer
        self.lhTargVar = tf.Variable(0.0, dtype=tf.float32, trainable=False) 
        self.lhTarg = tf.compat.v1.placeholder(tf.float32)
        self.updatelhTarg = tf.compat.v1.assign(self.lhTargVar, self.lhTarg)
        self.cost_reg_rnn = tf.constant(0., dtype=tf.float32)
        if config['l2_h'] > 0:
            self.hNorm = tf.reduce_mean(tf.square(self.states))
            self.cost_reg_rnn += tf.abs(self.hNorm - self.lhTargVar) * config['l2_h']
        else:
            self.hNorm = tf.reduce_mean(tf.square(self.states))

        # Get all trainable vars ready for gradient-based update
        self.var_list = tf.compat.v1.trainable_variables()
        self.full_var_list = [v for v in self.var_list]
        print([v.name for v in self.var_list])
        print([v.name for v in self.full_var_list])

        self.vName = []
        for v in self.full_var_list:
            name = v.name.split('/')[-1]
            name = name.split(':')[0]
            self.vName.append(name)

        # Input weight regularizer
        if config['l2_wI'] > 0:
            self.wNormI = tf.reduce_mean([tf.reduce_mean(tf.square(v)) for v in self.var_list if ('RNNin_weights' in v.name)])
            self.cost_reg_rnn += config['l2_wI'] * (self.wNormI)

        # Output weight regularizer
        if config['l2_wO'] > 0:
            self.wNormO = tf.reduce_mean([tf.reduce_mean(tf.square(v)) for v in self.var_list if ('out_RNN_weights' in v.name)])
            self.cost_reg_rnn += config['l2_wO'] * (self.wNormO)

        self.wNormR2 = tf.reduce_mean([tf.reduce_mean(tf.square(v)) for v in self.var_list if ('leakyRNN_weights' in v.name)])
        self.hMax   = tf.reduce_max(self.states)

        # Recurrent weight regularizer
        recWts = [v for v in self.var_list if ('leakyRNN_weights' in v.name)]
        self.sings = tf.linalg.svd(recWts[0], compute_uv=False)
        self.maxSingVal = tf.reduce_max(self.sings)
        self.topTenSings = self.sings[:10]
        if config['l2_wR'] > 0:
            self.wNormR = tf.reduce_mean(tf.square(self.sings[:int(config['svBnd'])]))
            self.cost_reg_rnn += config['l2_wR'] * (self.wNormR)#-self.lWRTargVar)

        # Setup Optimizer - ADAM
        self.opt_full = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate_full, beta1=0.3)
        cost_full = self.cost_lsq_rnn + self.cost_reg_rnn
        self.buildOpt(config, cost_full)

    # Backward pass
    def buildOpt(self, config, cost_full):
        self.full_grads_and_vars = self.opt_full.compute_gradients(cost_full, self.full_var_list)
        # gradient clipping
        vars = [x[1] for x in self.full_grads_and_vars]
        grads = [x[0] for x in self.full_grads_and_vars]
        clipped_grads, _ = tf.clip_by_global_norm(grads, 1)
        capped_gvs_full = zip(clipped_grads, vars)

        # Apply parameter updates
        self.optimizer_full = self.opt_full.apply_gradients(capped_gvs_full)

        # Re-init network state
        self.reinit_fullOpt = tf.compat.v1.initialize_variables(self.opt_full.variables())
        with tf.compat.v1.variable_scope("", reuse=True):
            self.initState = tf.compat.v1.get_variable("leakyRNN_init_state")
            Sup = tf.compat.v1.assign(self.initState, tf.nn.relu(self.initState))

        with tf.control_dependencies([Sup]):
            self.optimizer_full = tf.group([self.optimizer_full])

    # Initialize model
    def initialize(self, sess=None):
        assert self.sess is None
        if sess is None:
            sess = tf.compat.v1.get_default_session()
        self.sess = sess
        sess.run(tf.compat.v1.global_variables_initializer())

    # Restore trained model
    def restore(self, sCnt, sess=None):
        #assert self.sess is None
        #if sess is None:
        #    sess = tf.compat.v1.get_default_session()
        #self.sess = sess
        self.saver.restore(self.sess, os.path.join('data', self.config['save_name']+'_'+str(sCnt)+'.ckpt'))

    # Save model
    def save(self, sCnt):
        save_path = self.saver.save(self.sess, os.path.join('data', self.config['save_name']+'_'+str(sCnt)+'.ckpt'))
        print("Model saved in file: %s" % save_path)

    # Save ADAM's parameters
    def resetOpt(self, sess):
        sess.run([self.reinit_fullOpt])

    # Save all parameter values
    def getWeights(self, sess):
        wts = []
        wtNames = []
        for v in self.var_list:
            name = v.name.split('/')[-1]
            name = name.split(':')[0]
            wtNames.append(name)
            wts.append(sess.run([v])[0])

        return wts, wtNames

    # Homeostatic set point update
    def updateRegularizerTargets(self, hNorm, wRNorm, wINorm, sess):

        sess.run([self.updatelhTarg], feed_dict={self.lhTarg: hNorm})

        hN = sess.run([self.lhTargVar])
        print("New Reg Targets " + str(hN) + '(' + str(hNorm)+ ')') 

    # Change set of trainable parameters
    def removeTrainable(self, vNameList, config): 
        remFromFullVarList = list()
        for vName in vNameList:
            for vari in self.full_var_list:
                if vName in vari.name:
                    remFromFullVarList.append(vari)
        for rem in remFromFullVarList:
            print('Removing : ' + rem.name)
            self.full_var_list.remove(rem)
           
        print([v.name for v in self.var_list])
        print([v.name for v in self.full_var_list])

        cost_full = self.cost_lsq_rnn + self.cost_reg_rnn
        self.buildOpt(config, cost_full)

    # Get output weight matrix
    def getOutWeights(self, sess):
        outWeights = sess.run(self.w_rnn_out)
        return outWeights

    # Change weight matrix
    def setOutWeights(self, sess, newOutWeights):
        wtConst = tf.constant(newOutWeights, dtype = tf.float32)
        assign_op = self.w_rnn_out.assign(wtConst)
        sess.run(assign_op)

    # Print all trainable parameters
    def printTrainable(self):
        var_list = tf.compat.v1.trainable_variables()
        print('Trainable')
        print([v.name for v in var_list])
        
