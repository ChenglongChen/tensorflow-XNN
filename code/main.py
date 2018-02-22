"""

###
60 mins. (4 cores / 16 GB RAM / 60 minutes run-time / 1 GB scratch and output disk space)

###
I am running into the same problem described here and in #15026. The TensorFlow implementation of keras gives
very bad results in terms of loss and accuracy while the standalone keras performs as expected
https://github.com/tensorflow/tensorflow/issues/15831

Keras has much better gradients calculated than native TF
https://github.com/tensorflow/tensorflow/issues/13439


changing the kernel_initializer=tf.glorot_uniform_initializer() in dense
0.411 -> 0.417

sqrt(mean(log(y+1)^2)) = 3.07176
mean(log(y+1)^2) = 9.4357094976




# on one validation set
>>> np.mean(np.power(y_valid, 2))
9.439802942756756
>>> np.mean(np.power(y_pred, 2))
9.22734224590937

rmse(y_valid, y_pred)
0.40783908867824031

after normalization

def calibration(y_target, y_pred, m=None):
    if m is None:
        m = np.sqrt(np.mean(np.power(np.log1p(y_target), 2)))
    m = np.power(m, 2)
    y = np.np.log1p(y_pred), 2)


# ISSUE: validation error goes up epoch end

https://github.com/keras-team/keras/issues/5441

I hit something similar with Tensorflow, and it had to do with the dimensions
of the dimensions of the tensors going into the loss calculation. y_pred had
dimensions of (N,), while y_true had (N,1). The mean squared error was being
calculated on a NxN matrix, not an Nx1 or N-length vector, since y_pred-y_true
produces an NxN matrix. When I matched the dimensions, Tensorflow started working.
I can't seem to plumb through Keras though to see if it is the same issue.


THE FOLLOWING IS MORE RELEVANT
https://stats.stackexchange.com/questions/240371/training-performance-jumps-up-after-epoch-dev-performance-jumps-down
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import random_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops

import datetime
import os
import gc
import glob
import logging
import logging.handlers
import mmh3
import nltk
import numpy as np
import pandas as pd
import pickle as pkl
import shutil
import spacy
import time
import tensorflow as tf
import re
import string
import sys
from collections import Counter, defaultdict
from hashlib import md5

from fastcache import clru_cache as lru_cache

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk import ToktokTokenizer

from multiprocessing import Pool

try:
    import wordbatch
    from wordbatch.extractors import WordBag, WordHash
    from wordbatch.models import FTRL, FM_FTRL
except:
    pass

from six.moves import range
from six.moves import zip

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelBinarizer

from keras.preprocessing.sequence import pad_sequences as keras_pad_sequences


"""
https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0ahUKEwih7-6VlejYAhWGS98KHWeLCWQQFgg3MAE&url=https%3A%2F%2Fwww.bigdatarepublic.nl%2Fcustom-optimizer-in-tensorflow%2F&usg=AOvVaw3jmxRDqr2pkGRLvX6rNJrl
"""


class PowerSignOptimizer(optimizer.Optimizer):
    """Implementation of PowerSign.
    See [Bello et. al., 2017](https://arxiv.org/abs/1709.07417)
    @@__init__
    """

    def __init__(self, learning_rate=0.001, alpha=0.01, beta=0.5, use_locking=False, name="PowerSign"):
        super(PowerSignOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._alpha = alpha
        self._beta = beta

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._alpha_t = None
        self._beta_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._alpha_t = ops.convert_to_tensor(self._beta, name="alpha_t")
        self._beta_t = ops.convert_to_tensor(self._beta, name="beta_t")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)

        eps = 1e-7  # cap for moving average

        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta_t * m + eps, tf.abs(grad)))

        var_update = state_ops.assign_sub(var, lr_t * grad * tf.exp(
            tf.log(alpha_t) * tf.sign(grad) * tf.sign(m_t)))  # Update 'ref' by subtracting 'value
        # Create an op that groups multiple operations.
        # When this op finishes, all ops in input have finished
        return control_flow_ops.group(*[var_update, m_t])

    def _apply_sparse(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)

        eps = 1e-7  # cap for moving average

        m = self.get_slot(var, "m")
        m_slice = tf.gather(m, grad.indices)
        m_t = state_ops.scatter_update(m, grad.indices,
                                       tf.maximum(beta_t * m_slice + eps, tf.abs(grad.values)))
        m_t_slice = tf.gather(m_t, grad.indices)

        var_update = state_ops.scatter_sub(var, grad.indices, lr_t * grad.values * tf.exp(
            tf.log(alpha_t) * tf.sign(grad.values) * tf.sign(m_t_slice)))  # Update 'ref' by subtracting 'value
        # Create an op that groups multiple operations.
        # When this op finishes, all ops in input have finished
        return control_flow_ops.group(*[var_update, m_t])


class AddSignOptimizer(optimizer.Optimizer):
    """Implementation of AddSign.
    See [Bello et. al., 2017](https://arxiv.org/abs/1709.07417)
    @@__init__
    """

    def __init__(self, learning_rate=1.001, alpha=0.01, beta=0.5, use_locking=False, name="AddSign"):
        super(AddSignOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._alpha = alpha
        self._beta = beta

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._alpha_t = None
        self._beta_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._alpha_t = ops.convert_to_tensor(self._beta, name="beta_t")
        self._beta_t = ops.convert_to_tensor(self._beta, name="beta_t")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)
        alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)

        eps = 1e-7  # cap for moving average

        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta_t * m + eps, tf.abs(grad)))

        var_update = state_ops.assign_sub(var, lr_t * grad * (1.0 + alpha_t * tf.sign(grad) * tf.sign(m_t)))
        # Create an op that groups multiple operations
        # When this op finishes, all ops in input have finished
        return control_flow_ops.group(*[var_update, m_t])

    def _apply_sparse(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)
        alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)

        eps = 1e-7  # cap for moving average

        m = self.get_slot(var, "m")
        m_slice = tf.gather(m, grad.indices)
        m_t = state_ops.scatter_update(m, grad.indices,
                                       tf.maximum(beta_t * m_slice + eps, tf.abs(grad.values)))
        m_t_slice = tf.gather(m_t, grad.indices)

        var_update = state_ops.scatter_sub(var, grad.indices,
                                           lr_t * grad.values * (
                                                   1.0 + alpha_t * tf.sign(grad.values) * tf.sign(m_t_slice)))

        # Create an op that groups multiple operations
        # When this op finishes, all ops in input have finished
        return control_flow_ops.group(*[var_update, m_t])


class AMSGradOptimizer(optimizer.Optimizer):
    def __init__(self, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 use_locking=False, name="AMSGrad"):
        super(AMSGradOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "v_prime", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        # the following equations given in [1]
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_t = state_ops.assign(m, beta1_t * m + (1. - beta1_t) * grad, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_t = state_ops.assign(v, beta2_t * v + (1. - beta2_t) * tf.square(grad), use_locking=self._use_locking)
        v_prime = self.get_slot(var, "v_prime")
        v_t_prime = state_ops.assign(v_prime, tf.maximum(v_prime, v_t))

        var_update = state_ops.assign_sub(var,
                                          lr_t * m_t / (tf.sqrt(v_t_prime) + epsilon_t),
                                          use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, v_t, v_t_prime])

    # keras Nadam update rule
    def _apply_sparse(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        # the following equations given in [1]
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_t = state_ops.scatter_update(m, grad.indices,
                                       beta1_t * array_ops.gather(m, grad.indices) +
                                       (1. - beta1_t) * grad.values,
                                       use_locking=self._use_locking)
        m_t_slice = tf.gather(m_t, grad.indices)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_t = state_ops.scatter_update(v, grad.indices,
                                       beta2_t * array_ops.gather(v, grad.indices) +
                                       (1. - beta2_t) * tf.square(grad.values),
                                       use_locking=self._use_locking)
        v_prime = self.get_slot(var, "v_prime")
        v_t_slice = tf.gather(v_t, grad.indices)
        v_prime_slice = tf.gather(v_prime, grad.indices)
        v_t_prime = state_ops.scatter_update(v_prime, grad.indices, tf.maximum(v_prime_slice, v_t_slice))

        v_t_prime_slice = array_ops.gather(v_t_prime, grad.indices)
        var_update = state_ops.scatter_sub(var, grad.indices,
                                           lr_t * m_t_slice / (math_ops.sqrt(v_t_prime_slice) + epsilon_t),
                                           use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, v_t, v_t_prime])


class NadamOptimizer(optimizer.Optimizer):
    def __init__(self, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 schedule_decay=0.004, use_locking=False, name="Nadam"):
        super(NadamOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._schedule_decay = schedule_decay
        # momentum cache decay
        self._momentum_cache_decay = tf.cast(0.96, tf.float32)
        self._momentum_cache_const = tf.pow(self._momentum_cache_decay, 1. * schedule_decay)

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None
        self._schedule_decay_t = None

        # Variables to accumulate the powers of the beta parameters.
        # Created in _create_slots when we know the variables to optimize.
        self._beta1_power = None
        self._beta2_power = None
        self._iterations = None
        self._m_schedule = None

        # Created in SparseApply if needed.
        self._updated_lr = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")
        self._schedule_decay_t = ops.convert_to_tensor(self._schedule_decay, name="schedule_decay")

    def _create_slots(self, var_list):
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable. Sort the var_list to make sure this device is consistent across
        # workers (these need to go on the same PS, otherwise some updates are
        # silently ignored).
        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._iterations is None
        if not create_new and context.in_graph_mode():
            create_new = (self._iterations.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = variable_scope.variable(self._beta1,
                                                            name="beta1_power",
                                                            trainable=False)
                self._beta2_power = variable_scope.variable(self._beta2,
                                                            name="beta2_power",
                                                            trainable=False)
                self._iterations = variable_scope.variable(0.,
                                                           name="iterations",
                                                           trainable=False)
                self._m_schedule = variable_scope.variable(1.,
                                                           name="m_schedule",
                                                           trainable=False)
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _get_momentum_cache(self, schedule_decay_t, t):
        return tf.pow(self._momentum_cache_decay, t * schedule_decay_t)
        # return beta1_t * (1. - 0.5 * (tf.pow(self._momentum_cache_decay, t * schedule_decay_t)))

    """very slow
    we simply use the nadam update rule without warming momentum schedule
    def _apply_dense(self, grad, var):
        t = math_ops.cast(self._iterations, var.dtype.base_dtype) + 1.
        m_schedule = math_ops.cast(self._m_schedule, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        schedule_decay_t = math_ops.cast(self._schedule_decay_t, var.dtype.base_dtype)

        # Due to the recommendations in [2], i.e. warming momentum schedule
        # see keras Nadam
        momentum_cache_t = self._get_momentum_cache(beta1_t, schedule_decay_t, t)
        momentum_cache_t_1 = self._get_momentum_cache(beta1_t, schedule_decay_t, t+1.)
        m_schedule_new = m_schedule * momentum_cache_t
        m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1

        # the following equations given in [1]
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_t = state_ops.assign(m, beta1_t * m + (1. - beta1_t) * grad, use_locking=self._use_locking)
        g_prime = grad / (1. - m_schedule_new)
        m_t_prime = m_t / (1. - m_schedule_next)
        m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_t = state_ops.assign(v, beta2_t * v + (1. - beta2_t) * tf.square(grad), use_locking=self._use_locking)
        v_t_prime = v_t / (1. - tf.pow(beta2_t, t))

        var_update = state_ops.assign_sub(var,
                                      lr_t * m_t_bar / (tf.sqrt(v_t_prime) + epsilon_t),
                                      use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, v_t])
    """

    # nadam update rule without warming momentum schedule
    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        return training_ops.apply_adam(
            var,
            m,
            v,
            math_ops.cast(self._beta1_power, var.dtype.base_dtype),
            math_ops.cast(self._beta2_power, var.dtype.base_dtype),
            math_ops.cast(self._lr_t, var.dtype.base_dtype),
            math_ops.cast(self._beta1_t, var.dtype.base_dtype),
            math_ops.cast(self._beta2_t, var.dtype.base_dtype),
            math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
            grad,
            use_locking=self._use_locking,
            use_nesterov=True).op

    def _resource_apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        return training_ops.resource_apply_adam(
            var.handle,
            m.handle,
            v.handle,
            math_ops.cast(self._beta1_power, grad.dtype.base_dtype),
            math_ops.cast(self._beta2_power, grad.dtype.base_dtype),
            math_ops.cast(self._lr_t, grad.dtype.base_dtype),
            math_ops.cast(self._beta1_t, grad.dtype.base_dtype),
            math_ops.cast(self._beta2_t, grad.dtype.base_dtype),
            math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
            grad,
            use_locking=self._use_locking,
            use_nesterov=True)

    # keras Nadam update rule
    def _apply_sparse(self, grad, var):
        t = math_ops.cast(self._iterations, var.dtype.base_dtype) + 1.
        m_schedule = math_ops.cast(self._m_schedule, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        schedule_decay_t = math_ops.cast(self._schedule_decay_t, var.dtype.base_dtype)

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_power = self._get_momentum_cache(schedule_decay_t, t)
        momentum_cache_t = beta1_t * (1. - 0.5 * momentum_cache_power)
        momentum_cache_t_1 = beta1_t * (1. - 0.5 * momentum_cache_power * self._momentum_cache_const)
        m_schedule_new = m_schedule * momentum_cache_t
        m_schedule_next = m_schedule_new * momentum_cache_t_1

        # the following equations given in [1]
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_t = state_ops.scatter_update(m, grad.indices,
                                       beta1_t * array_ops.gather(m, grad.indices) +
                                       (1. - beta1_t) * grad.values,
                                       use_locking=self._use_locking)
        g_prime_slice = grad.values / (1. - m_schedule_new)
        m_t_prime_slice = array_ops.gather(m_t, grad.indices) / (1. - m_schedule_next)
        m_t_bar_slice = (1. - momentum_cache_t) * g_prime_slice + momentum_cache_t_1 * m_t_prime_slice

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_t = state_ops.scatter_update(v, grad.indices,
                                       beta2_t * array_ops.gather(v, grad.indices) +
                                       (1. - beta2_t) * tf.square(grad.values),
                                       use_locking=self._use_locking)
        v_t_prime_slice = array_ops.gather(v_t, grad.indices) / (1. - tf.pow(beta2_t, t))

        var_update = state_ops.scatter_sub(var, grad.indices,
                                           lr_t * m_t_bar_slice / (math_ops.sqrt(v_t_prime_slice) + epsilon_t),
                                           use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._iterations):
                update_beta1 = self._beta1_power.assign(
                    self._beta1_power * self._beta1_t,
                    use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(
                    self._beta2_power * self._beta2_t,
                    use_locking=self._use_locking)
                t = self._iterations + 1.
                update_iterations = self._iterations.assign(t, use_locking=self._use_locking)
                momentum_cache_power = self._get_momentum_cache(self._schedule_decay_t, t)
                momentum_cache_t = self._beta1_t * (1. - 0.5 * momentum_cache_power)
                update_m_schedule = self._m_schedule.assign(
                    self._m_schedule * momentum_cache_t,
                    use_locking=self._use_locking)
        return control_flow_ops.group(
            *update_ops + [update_beta1, update_beta2] + [update_iterations, update_m_schedule],
            name=name_scope)


##############################################################################
def _timestamp():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d%H%M")
    return now_str


def _get_logger(logdir, logname, loglevel=logging.INFO):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    formatter = logging.Formatter(fmt)

    handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(logdir, logname),
        maxBytes=2 * 1024 * 1024 * 1024,
        backupCount=10)
    handler.setFormatter(formatter)

    logger = logging.getLogger("")
    logger.addHandler(handler)
    logger.setLevel(loglevel)
    return logger


def _makedirs(dir, force=False):
    if os.path.exists(dir):
        if force:
            shutil.rmtree(dir)
            os.makedirs(dir)
    else:
        os.makedirs(dir)


_makedirs("./log")
logger = _get_logger("./log", "hyperopt-%s.log" % _timestamp())

##############################################################################

RUNNING_MODE = "validation"
# RUNNING_MODE = "submission"
DEBUG = False
DUMP_DATA = True
USE_PREPROCESSED_DATA = True

USE_MULTITHREAD = False
if RUNNING_MODE == "submission":
    N_JOBS = 4
else:
    N_JOBS = 4
NUM_PARTITIONS = 32

DEBUG_SAMPLE_NUM = 200000
LRU_MAXSIZE = 2 ** 16

#######################################
# File
MISSING_VALUE_STRING = "MISSINGVALUE"
DROP_ZERO_PRICE = True
#######################################


# Preprocessing
USE_SPACY_TOKENIZER = False
USE_NLTK_TOKENIZER = False
USE_KAGGLE_TOKENIZER = False
# default: '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
KERAS_TOKENIZER_FILTERS = '\'!"#%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n'
KERAS_TOKENIZER_FILTERS = ""
KERAS_SPLIT = " "

USE_LEMMATIZER = False
USE_STEMMER = False
USE_CLEAN = True

WORDREPLACER_DICT = {
    "bnwt": "brand new with tags",
    "nwt": "new with tags",
    "bnwot": "brand new without tags",
    "nwot": "new without tags",
    "bnip": "brand new in packet",
    "nip": "new in packet",
    "bnib": "brand new in box",
    "nib": "new in box",
    "mib": "mint in box",
    "mwob": "mint without box",
    "mip": "mint in packet",
    "mwop": "mint without packet"
}

BRAND_NAME_PATTERN_LIST = [
    ("nike", "nike"),
    ("pink", "pink"),
    ("apple", "iphone|ipod|ipad|iwatch|apple|mac"),
    ("victoria's secret", "victoria"),
    ("lularoe", "lularoe"),
    ("nintendo", "nintendo"),
    ("lululemon", "lululemon"),
    ("forever 21", "forever\s+21|forever\s+twenty\s+one"),
    ("michael kors", "michael\s+kors"),
    ("american eagle", "american\s+eagle"),
    ("rae dunn", "rae dunn"),
]

# word count |   #word
#    >= 1    |  195523
#    >= 2    |   93637
#    >= 3    |   67498
#    >= 4    |   56265
#    >= 5    |   49356
MAX_NUM_WORDS = 80000
MAX_NUM_BIGRAMS = 50000
MAX_NUM_TRIGRAMS = 50000
MAX_NUM_SUBWORDS = 20000
MAX_NUM_SUBWORDS_LIST = 50000

NUM_TOP_WORDS_NAME = 50
NUM_TOP_WORDS_ITEM_DESC = 50

global MAX_NUM_BRANDS
global MAX_NUM_CATEGORIES
global MAX_NUM_CATEGORIES_LST
global MAX_NUM_CONDITIONS
global MAX_NUM_SHIPPINGS
global target_scaler

MAX_CATEGORY_NAME_LEN = 3

EXTRACTED_BIGRAM = True
EXTRACTED_TRIGRAM = True
EXTRACTED_SUBWORD = False
EXTRACTED_SUBWORD_LIST = False
VOCAB_HASHING_TRICK = False

######################
ENABLE_STACKING = False

####################################################################
HYPEROPT_MAX_EVALS = 500
HYPEROPT_MAX_EVALS = 1

param_space_com = {
    "model_dir": "./weights",
    # size for the attention block
    "item_condition_size": 5,
    "shipping_size": 1,
    "num_vars_size": 3,
    # pad_sequences
    "pad_sequences_padding": "post",
    "pad_sequences_truncating": "post",
    # optimization
    "optimizer_clipnorm": 1.,
    "batch_size_train": 512,
    "batch_size_inference": 512*2,
    "shuffle_with_replacement": False,
    # CyclicLR
    "t_mul": 1,
    "snapshot_every_num_cycle": 128,
    "max_snapshot_num": 14,
    "snapshot_every_epoch": 4,  # for t_mult != 1
    "eval_every_num_update": 1000,
    # static param
    "random_seed": 2018,
    "n_folds": 1,
    "validation_ratio": 0.4,
}

param_space_best = {

    #### params for input
    # bigram/trigram/subword
    "use_bigram": True,
    "use_trigram": True,
    "use_subword": False,
    "use_subword_list": False,

    # seq len
    "max_sequence_length_name": 10,
    "max_sequence_length_item_desc": 50,
    "max_sequence_length_category_name": 10,
    "max_sequence_length_item_desc_subword": 45,

    #### params for embed
    "embedding_dim": 250,
    "embedding_dropout": 0.,
    "embedding_mask_zero": False,
    "embedding_mask_zero_subword": False,

    #### params for encode
    "encode_method": "fasttext",
    # cnn
    "cnn_num_filters": 16,
    "cnn_filter_sizes": [2, 3],
    "cnn_timedistributed": False,
    # rnn
    "rnn_num_units": 16,
    "rnn_cell_type": "gru",
    #### params for attend
    "attend_method": "ave",

    #### params for predict
    # deep
    "enable_deep": True,
    # fm
    "enable_fm_first_order": True,
    "enable_fm_second_order": True,
    "enable_fm_higher_order": False,
    # fc block
    "fc_type": "fc",
    "fc_dim": 64,
    "fc_dropout": 0.,

    #### params for optimization
    "optimizer_type": "nadam",  # "nadam",  # ""lazyadam", "nadam"
    "max_lr_exp": 0.005,
    "lr_decay_each_epoch_exp": 0.9,
    "lr_jump_exp": True,
    "max_lr_cosine": 0.005,
    "base_lr": 0.00001,  # minimum lr
    "lr_decay_each_epoch_cosine": 0.5,
    "lr_jump_rate": 1.,
    "snapshot_before_restarts": 4,
    "beta1": 0.975,
    "beta2": 0.999,
    "schedule_decay": 0.004,
    # "lr_schedule": "exponential_decay",
    "lr_schedule": "cosine_decay_restarts",
    "epoch": 4,
    # CyclicLR
    "num_cycle_each_epoch": 8,

    #### params ensemble
    "enable_stacking": False,
    "enable_snapshot_ensemble": True,
    "n_runs": 2,

}
param_space_best.update(param_space_com)
if RUNNING_MODE == "submission":
    EXTRACTED_BIGRAM = param_space_best["use_bigram"]
    EXTRACTED_SUBWORD = param_space_best["use_subword"]

param_space_hyperopt = param_space_best

int_params = [
    "max_sequence_length_name",
    "max_sequence_length_item_desc",
    "max_sequence_length_item_desc_subword",
    "max_sequence_length_category_name",
    "embedding_dim", "embedding_dim",
    "cnn_num_filters", "rnn_num_units", "fc_dim",
    "epoch", "n_runs",
    "num_cycle_each_epoch", "t_mul", "snapshot_every_num_cycle",
]
int_params = set(int_params)

if DEBUG:
    param_space_hyperopt["num_cycle_each_epoch"] = param_space_best["num_cycle_each_epoch"] = 2
    param_space_hyperopt["snapshot_every_num_cycle"] = param_space_best["snapshot_every_num_cycle"] = 1
    param_space_hyperopt["batch_size_train"] = param_space_best["batch_size_train"] = 512
    param_space_hyperopt["batch_size_inference"] = param_space_best["batch_size_inference"] = 512


####################################################################################################
########################################### NLP ####################################################
####################################################################################################
def mmh3_hash_function(x):
    return mmh3.hash(x, 42, signed=True)


def md5_hash_function(x):
    return int(md5(x.encode()).hexdigest(), 16)


@lru_cache(LRU_MAXSIZE)
def hashing_trick(string, n, hash_function="mmh3"):
    if hash_function == "mmh3":
        hash_function = mmh3_hash_function
    elif hash_function == "md5":
        hash_function = md5_hash_function
    i = (hash_function(string) % n) + 1
    return i


# 5.67 µs ± 78.2 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
@lru_cache(LRU_MAXSIZE)
def get_subword_for_word_all(word, n1=3, n2=6):
    z = []
    z_append = z.append
    word = "*" + word + "*"
    l = len(word)
    z_append(word)
    for k in range(n1, n2 + 1):
        for i in range(l - k + 1):
            z_append(word[i:i + k])
    return z


@lru_cache(LRU_MAXSIZE)
def get_subword_for_word_all0(word, n1=3, n2=6):
    z = []
    z_append = z.append
    word = "*" + word + "*"
    l = len(word)
    z_append(word)
    if l > n1:
        n2 = min(n2, l - 1)
        for i in range(l - n1 + 1):
            for k in range(n1, n2 + 1):
                if 2 * i + n2 < l:
                    z_append(word[i:(i + k)])
                    if i == 0:
                        z_append(word[-(i + k + 1):])
                    else:
                        z_append(word[-(i + k + 1):-i])
                else:
                    if 2 * i + k < l:
                        z_append(word[i:(i + k)])
                        z_append(word[-(i + k + 1):-i])
                    elif 2 * (i - 1) + n2 < l:
                        z_append(word[i:(i + k)])
    return z


# 3.44 µs ± 101 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
@lru_cache(LRU_MAXSIZE)
def get_subword_for_word0(word, n1=4, n2=5, include_self=False):
    """only extract the prefix and suffix"""
    l = len(word)
    n1 = min(n1, l)
    n2 = min(n2, l)
    z1 = [word[:k] for k in range(n1, n2 + 1)]
    z2 = [word[-k:] for k in range(n1, n2 + 1)]
    z = z1 + z2
    if include_self:
        z.append(word)
    return z


# 2.49 µs ± 104 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
@lru_cache(LRU_MAXSIZE)
def get_subword_for_word(word, n1=3, n2=6, include_self=False):
    """only extract the prefix and suffix"""
    z = []
    if len(word) >= n1:
        word = "*" + word + "*"
        l = len(word)
        n1 = min(n1, l)
        n2 = min(n2, l)
        # bind method outside of loop to reduce overhead
        # https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/feature_extraction/text.py#L144
        z_append = z.append
        if include_self:
            z_append(word)
        for k in range(n1, n2 + 1):
            z_append(word[:k])
            z_append(word[-k:])
    return z


# 564 µs ± 14.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
def get_subword_for_list0(input_list, n1=4, n2=5):
    subword_lst = [get_subword_for_word(w, n1, n2) for w in input_list]
    subword_lst = [w for ws in subword_lst for w in ws]
    return subword_lst


# 505 µs ± 15 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
def get_subword_for_list(input_list, n1=4, n2=5):
    subwords = []
    subwords_extend = subwords.extend
    for w in input_list:
        subwords_extend(get_subword_for_word(w, n1, n2))
    return subwords


def get_subword_list_for_list(input_list, n1=3, n2=7):
    subwords = []
    subwords_append = subwords.append
    for w in input_list:
        subwords_append(get_subword_for_word(w, n1, n2))
    return subwords


@lru_cache(LRU_MAXSIZE)
def get_subword_for_text(text, n1=4, n2=5):
    return get_subword_for_list(text.split(" "), n1, n2)


stopwords = [
    "and",
    "the",
    "for",
    "a",
    "in",
    "to",
    "is",
    # "s",
    "of",
    "i",
    "on",
    "it",
    "you",
    "your",
    "are",
    "this",
    "my",
]
stopwords = set(stopwords)


# spacy model
class SpacyTokenizer(object):
    def __init__(self):
        self.nlp = spacy.load("en", disable=["parser", "tagger", "ner"])

    def tokenize(self, text):
        tokens = [tok.lower_ for tok in self.nlp(text)]
        # tokens = get_valid_words(tokens)
        return tokens


LEMMATIZER = nltk.stem.wordnet.WordNetLemmatizer()
STEMMER = nltk.stem.snowball.EnglishStemmer()
TOKTOKTOKENIZER = ToktokTokenizer()


# SPACYTOKENIZER = SpacyTokenizer()


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def get_valid_words(sentence):
    res = [w.strip() for w in sentence]
    return [w for w in res if w]


@lru_cache(LRU_MAXSIZE)
def stem_word(word):
    return STEMMER.stem(word)


@lru_cache(LRU_MAXSIZE)
def lemmatize_word(word, pos=wordnet.NOUN):
    return LEMMATIZER.lemmatize(word, pos)


def stem_sentence(sentence):
    return [stem_word(w) for w in get_valid_words(sentence)]


def lemmatize_sentence(sentence):
    res = []
    sentence_ = get_valid_words(sentence)
    for word, pos in pos_tag(sentence_):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatize_word(word, pos=wordnet_pos))
    return res


def stem_lemmatize_sentence(sentence):
    return [stem_word(word) for word in lemmatize_sentence(sentence)]


TRANSLATE_MAP = maketrans(KERAS_TOKENIZER_FILTERS, KERAS_SPLIT * len(KERAS_TOKENIZER_FILTERS))


def get_tokenizer():
    if USE_LEMMATIZER and USE_STEMMER:
        return stem_lemmatize_sentence
    elif USE_LEMMATIZER:
        return lemmatize_sentence
    elif USE_STEMMER:
        return stem_sentence
    else:
        return get_valid_words


tokenizer = get_tokenizer()


#
# https://stackoverflow.com/questions/21883108/fast-optimize-n-gram-implementations-in-python
# @lru_cache(LRU_MAXSIZE)
# 40.1 µs ± 918 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
def get_ngrams0(words, ngram_value):
    # # return list
    ngrams = [" ".join(ngram) for ngram in zip(*[words[i:] for i in range(ngram_value)])]
    # return generator (10x faster)
    # ngrams = (" ".join(ngram) for ngram in zip(*[words[i:] for i in range(ngram_value)]))
    return ngrams


# 36.2 µs ± 1.04 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
def get_ngrams(words, ngram_value):
    tokens = []
    tokens_append = tokens.append
    for i in range(ngram_value):
        tokens_append(words[i:])
    ngrams = []
    ngrams_append = ngrams.append
    space_join = " ".join
    for ngram in zip(*tokens):
        ngrams_append(space_join(ngram))
    return ngrams


def get_bigrams(words):
    return get_ngrams(words, 2)


def get_trigrams(words):
    return get_ngrams(words, 3)


@lru_cache(LRU_MAXSIZE)
# 68.8 µs ± 1.86 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
def get_ngrams_range(text, ngram_range):
    unigrams = text.split(" ")
    ngrams = []
    ngrams_extend = ngrams.extend
    for i in range(ngram_range[0], ngram_range[1] + 1):
        ngrams_extend(get_ngrams(unigrams, i))
    return ngrams


# 69.6 µs ± 1.45 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
def get_ngrams_range0(text, ngram_range):
    unigrams = text.split(" ")
    res = []
    for i in ngram_range:
        res += get_ngrams(unigrams, i)
    res += unigrams
    return res


@lru_cache(LRU_MAXSIZE)
def stem(s):
    return STEMMER.stem(s)


tags = re.compile(r'<.+?>')
whitespace = re.compile(r'\s+')
non_letter = re.compile(r'\W+')


@lru_cache(LRU_MAXSIZE)
def clean_text(text):
    # text = text.lower()
    text = non_letter.sub(' ', text)

    tokens = []

    for t in text.split():
        # if len(t) <= 2 and not t.isdigit():
        #     continue
        if t in stopwords:
            continue
        t = stem(t)
        tokens.append(t)

    text = ' '.join(tokens)

    text = whitespace.sub(' ', text)
    text = text.strip()
    return text.split(" ")


@lru_cache(LRU_MAXSIZE)
def tokenize(text):
    if USE_NLTK_TOKENIZER:
        # words = get_valid_words(word_tokenize(text))
        # words = get_valid_words(wordpunct_tokenize(text))
        words = get_valid_words(TOKTOKTOKENIZER.tokenize(text))
    elif USE_SPACY_TOKENIZER:
        words = get_valid_words(SPACYTOKENIZER.tokenize(text))
    elif USE_KAGGLE_TOKENIZER:
        words = clean_text(text)
    else:
        words = tokenizer(text.translate(TRANSLATE_MAP).split(KERAS_SPLIT))
    return words


@lru_cache(LRU_MAXSIZE)
def tokenize_with_subword(text, n1=4, n2=5):
    words = tokenize(text)
    subwords = get_subword_for_list(words, n1, n2)
    return words + subwords


######################################################################
# --------------------------- Processor ---------------------------
## base class
## Most of the processings can be casted into the "pattern-replace" framework
class BaseReplacer:
    def __init__(self, pattern_replace_pair_list=[]):
        self.pattern_replace_pair_list = pattern_replace_pair_list


## deal with word replacement
# 1st solution in CrowdFlower
class WordReplacer(BaseReplacer):
    def __init__(self, replace_dict):
        self.replace_dict = replace_dict
        self.pattern_replace_pair_list = []
        for k, v in self.replace_dict.items():
            # pattern = r"(?<=\W|^)%s(?=\W|$)" % k
            pattern = k
            replace = v
            self.pattern_replace_pair_list.append((pattern, replace))


class MerCariCleaner(BaseReplacer):
    """https://stackoverflow.com/questions/7317043/regex-not-operator
    """

    def __init__(self):
        self.pattern_replace_pair_list = [
            # # remove filters
            # (r'[-!\'\"#&()\*\+,-/:;<=＝>?@\[\\\]^_`{|}~\t\n]+', r""),
            # remove punctuation ".", e.g.,
            (r"(?<!\d)\.(?!\d+)", r" "),
            # iphone 6/6s -> iphone 6 / 6s
            # iphone 6:6s -> iphone 6 : 6s
            (r"(\W+)", r" \1 "),
            # # non
            # (r"[^A-Za-z0-9]+", r" "),
            # 6s -> 6 s
            # 32gb -> 32 gb
            # 4oz -> 4 oz
            # 4pcs -> 4 pcs
            (r"(\d+)([a-zA-Z])", r"\1 \2"),
            # iphone5 -> iphone 5
            # xbox360 -> xbox 360
            # only split those with chars length > 3
            (r"([a-zA-Z]{3,})(\d+)", r"\1 \2"),
        ]


###########################################
def df_lower(df):
    return df.str.lower()


def df_contains(df, pat):
    return df.str.contains(pat).astype(int)


def df_len(df):
    return df.str.len().astype(float)


def df_num_tokens(df):
    return df.str.split().apply(len).astype(float)


def df_in(df, col1, col2):
    def _in(x):
        return x[col1] in x[col2]

    return df.apply(_in, 1).astype(int)


def df_brand_in_name(df):
    return df_in(df, "brand_name", "name")


def df_category1_in_name(df):
    return df_in(df, "category_name1", "name")


def df_category2_in_name(df):
    return df_in(df, "category_name2", "name")


def df_category3_in_name(df):
    return df_in(df, "category_name3", "name")


def df_brand_in_desc(df):
    return df_in(df, "brand_name", "item_desc")


def df_category1_in_desc(df):
    return df_in(df, "category_name1", "item_desc")


def df_category2_in_desc(df):
    return df_in(df, "category_name2", "item_desc")


def df_category3_in_desc(df):
    return df_in(df, "category_name3", "item_desc")


def df_clean(df):
    for pat, repl in MerCariCleaner().pattern_replace_pair_list:
        df = df.str.replace(pat, repl)
    # for pat, repl in WordReplacer(WORDREPLACER_DICT).pattern_replace_pair_list:
    #     df = df.str.replace(pat, repl)
    return df


def df_tokenize(df):
    return df.apply(tokenize)


def df_tokenize_with_subword(df):
    return df.apply(tokenize_with_subword)


def df_get_bigram(df):
    return df.apply(get_bigrams)


def df_get_trigram(df):
    return df.apply(get_trigrams)


def df_get_subword(df):
    return df.apply(get_subword_for_list)


def df_get_subword_list(df):
    """
    :param df:
    :return: a list of words. a word is a list of subwords
    """
    return df.apply(get_subword_list_for_list)


def parallelize_df_func(df, func, num_partitions=NUM_PARTITIONS, n_jobs=N_JOBS):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(n_jobs)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


######################################################################

def load_train_data():
    types_dict_train = {
        'train_id': 'int32',
        'item_condition_id': 'int32',
        'price': 'float32',
        'shipping': 'int8',
        'name': 'str',
        'brand_name': 'str',
        'item_desc': 'str',
        'category_name': 'str',
    }
    df = pd.read_csv('../input/train.tsv', delimiter='\t', low_memory=True, dtype=types_dict_train)
    df.rename(columns={"train_id": "id"}, inplace=True)
    df.rename(columns={"item_description": "item_desc"}, inplace=True)
    if DROP_ZERO_PRICE:
        df = df[df.price > 0].copy()
    price = np.log1p(df.price.values)
    df.drop("price", axis=1, inplace=True)
    df["price"] = price
    df["is_train"] = 1
    df["missing_brand_name"] = df["brand_name"].isnull().astype(int)
    df["missing_category_name"] = df["category_name"].isnull().astype(int)
    missing_ind = np.logical_or(df["item_desc"].isnull(),
                                df["item_desc"].str.lower().str.contains("no\s+description\s+yet"))
    df["missing_item_desc"] = missing_ind.astype(int)
    df["item_desc"][missing_ind] = df["name"][missing_ind]
    gc.collect()
    if DEBUG:
        return df.head(DEBUG_SAMPLE_NUM)
    else:
        return df


def load_test_data(chunksize=350000*2):
    types_dict_test = {
        'test_id': 'int32',
        'item_condition_id': 'int32',
        'shipping': 'int8',
        'name': 'str',
        'brand_name': 'str',
        'item_description': 'str',
        'category_name': 'str',
    }
    chunks = pd.read_csv('../input/test.tsv', delimiter='\t',
                         low_memory=True, dtype=types_dict_test,
                         chunksize=chunksize)
    for df in chunks:
        df.rename(columns={"test_id": "id"}, inplace=True)
        df.rename(columns={"item_description": "item_desc"}, inplace=True)
        df["missing_brand_name"] = df["brand_name"].isnull().astype(int)
        df["missing_category_name"] = df["category_name"].isnull().astype(int)
        missing_ind = np.logical_or(df["item_desc"].isnull(),
                                    df["item_desc"].str.lower().str.contains("no\s+description\s+yet"))
        df["missing_item_desc"] = missing_ind.astype(int)
        df["item_desc"][missing_ind] = df["name"][missing_ind]
        yield df


@lru_cache(1024)
def split_category_name(row):
    grps = row.split("/")
    if len(grps) > MAX_CATEGORY_NAME_LEN:
        grps = grps[:MAX_CATEGORY_NAME_LEN]
    else:
        grps += [MISSING_VALUE_STRING.lower()] * (MAX_CATEGORY_NAME_LEN - len(grps))
    return tuple(grps)


"""
https://stackoverflow.com/questions/3172173/most-efficient-way-to-calculate-frequency-of-values-in-a-python-list

| approach       | american-english, |      big.txt, | time w.r.t. defaultdict |
|                |     time, seconds | time, seconds |                         |
|----------------+-------------------+---------------+-------------------------|
| Counter        |             0.451 |         3.367 |                     3.6 |
| setdefault     |             0.348 |         2.320 |                     2.5 |
| list           |             0.277 |         1.822 |                       2 |
| try/except     |             0.158 |         1.068 |                     1.2 |
| defaultdict    |             0.141 |         0.925 |                       1 |
| numpy          |             0.012 |         0.076 |                   0.082 |
| S.Mark's ext.  |             0.003 |         0.019 |                   0.021 |
| ext. in Cython |             0.001 |         0.008 |                  0.0086 |

code: https://gist.github.com/347000
"""


def get_word_index0(words, max_num, prefix):
    word_counts = defaultdict(int)
    for ws in words:
        for w in ws:
            word_counts[w] += 1
    wcounts = list(word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    sorted_voc = [wc[0] for wc in wcounts[:(max_num - 1)]]
    word_index = dict(zip(sorted_voc, range(2, max_num)))
    return word_index


def get_word_index1(words, max_num, prefix):
    word_counts = Counter([w for ws in words for w in ws])
    # """
    wcounts = list(word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    sorted_voc = [wc[0] for wc in wcounts]
    del wcounts
    gc.collect()
    # only keep MAX_NUM_WORDS
    sorted_voc = sorted_voc[:(max_num - 1)]
    # note that index 0 is reserved, never assigned to an existing word
    word_index = dict(zip(sorted_voc, range(2, max_num)))
    return word_index


def get_word_index(words, max_num, prefix):
    word_counts = Counter([w for ws in words for w in ws])
    sorted_voc = [w for w, c in word_counts.most_common(max_num - 1)]
    word_index = dict(zip(sorted_voc, range(2, max_num)))
    return word_index


# Bucket Sort
# Time:  O(n + klogk) ~ O(n + nlogn)
# Space: O(n)
class BucketSort(object):
    def topKFrequent(self, words, k):
        counts = defaultdict(int)
        for ws in words:
            for w in ws:
                counts[w] += 1

        buckets = [[]] * (sum(counts.values()) + 1)
        for i, count in counts.items():
            buckets[count].append(i)

        result = []
        # result_append = result.append
        for i in reversed(range(len(buckets))):
            for j in range(len(buckets[i])):
                # slower
                # result_append(buckets[i][j])
                result.append(buckets[i][j])
                if len(result) == k:
                    return result
        return result


# Quick Select
# Time:  O(n) ~ O(n^2), O(n) on average.
# Space: O(n)
from random import randint


class QuickSelect(object):
    def topKFrequent(self, words, k):
        """
        :type words: List[str]
        :type k: int
        :rtype: List[str]
        """
        counts = defaultdict(int)
        for ws in words:
            for w in ws:
                counts[w] += 1
        p = []
        for key, val in counts.items():
            p.append((-val, key))
        self.kthElement(p, k)

        result = []
        sorted_p = sorted(p[:k])
        for i in range(k):
            result.append(sorted_p[i][1])
        return result

    def kthElement(self, nums, k):  # O(n) on average
        def PartitionAroundPivot(left, right, pivot_idx, nums):
            pivot_value = nums[pivot_idx]
            new_pivot_idx = left
            nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
            for i in range(left, right):
                if nums[i] < pivot_value:
                    nums[i], nums[new_pivot_idx] = nums[new_pivot_idx], nums[i]
                    new_pivot_idx += 1

            nums[right], nums[new_pivot_idx] = nums[new_pivot_idx], nums[right]
            return new_pivot_idx

        left, right = 0, len(nums) - 1
        while left <= right:
            pivot_idx = randint(left, right)
            new_pivot_idx = PartitionAroundPivot(left, right, pivot_idx, nums)
            if new_pivot_idx == k - 1:
                return
            elif new_pivot_idx > k - 1:
                right = new_pivot_idx - 1
            else:  # new_pivot_idx < k - 1.
                left = new_pivot_idx + 1


# top_k_selector = BucketSort()


top_k_selector = QuickSelect()


def get_word_index(words, max_num, prefix):
    sorted_voc = top_k_selector.topKFrequent(words, max_num - 1)
    word_index = dict(zip(sorted_voc, range(2, max_num)))
    return word_index


class MyLabelEncoder(object):
    """safely handle unknown label"""

    def __init__(self):
        self.mapper = {}

    def fit(self, X):
        uniq_X = np.unique(X)
        # reserve 0 for unknown
        self.mapper = dict(zip(uniq_X, range(1, len(uniq_X) + 1)))
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _map(self, x):
        return self.mapper.get(x, 0)

    def transform(self, X):
        return list(map(self._map, X))


class MyStandardScaler(object):
    def __init__(self, identity=False, epsilon=1e-8):
        self.identity = identity
        self.mean_ = 0.
        self.scale_ = 1.
        self.epsilon = epsilon

    def fit(self, X):
        if not self.identity:
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
        else:
            self.epsilon = 0.

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        return (X - self.mean_) / (self.scale_ + self.epsilon)

    def inverse_transform(self, X):
        return X * (self.scale_ + self.epsilon) + self.mean_


def preprocess(df, word_index=None, bigram_index=None,
               trigram_index=None, subword_index=None,
               subword_list_index=None, label_encoder=None):
    start_time = time.time()

    #### fill na
    df.fillna(MISSING_VALUE_STRING, inplace=True)
    gc.collect()

    #### to lower case
    df["name"] = df.name.str.lower()
    df["brand_name"] = df.brand_name.str.lower()
    df["category_name"] = df.category_name.str.lower()
    df["item_desc"] = df.item_desc.str.lower()
    gc.collect()
    print("[%.5f] Done df_lower" % (time.time() - start_time))

    #### split category name
    for i, cat in enumerate(zip(*df.category_name.apply(split_category_name))):
        df["category_name%d" % (i + 1)] = cat
        gc.collect()

    #### regex based cleaning
    if USE_CLEAN:
        df["name"] = parallelize_df_func(df["name"], df_clean)
        df["item_desc"] = parallelize_df_func(df["item_desc"], df_clean)
        # df["category_name"] = parallelize_df_func(df["category_name"], df_clean)
        print("[%.5f] Done df_clean" % (time.time() - start_time))
        gc.collect()

    #### tokenize
    # print("   Fitting tokenizer...")
    df["seq_name"] = parallelize_df_func(df["name"], df_tokenize)
    df["seq_item_desc"] = parallelize_df_func(df["item_desc"], df_tokenize)
    # df["seq_brand_name"] = parallelize_df_func(df["brand_name"], df_tokenize)
    # df["seq_category_name"] = parallelize_df_func(df["category_name"], df_tokenize)
    gc.collect()
    print("[%.5f] Done df_tokenize" % (time.time() - start_time))
    df.drop(["name"], axis=1, inplace=True)
    df.drop(["item_desc"], axis=1, inplace=True)
    gc.collect()
    if USE_MULTITHREAD:
        if EXTRACTED_BIGRAM:
            df["seq_bigram_item_desc"] = parallelize_df_func(df["seq_item_desc"], df_get_bigram)
            print("[%.5f] Done df_get_bigram" % (time.time() - start_time))
        if EXTRACTED_TRIGRAM:
            df["seq_trigram_item_desc"] = parallelize_df_func(df["seq_item_desc"], df_get_trigram)
            print("[%.5f] Done df_get_trigram" % (time.time() - start_time))
        if EXTRACTED_SUBWORD:
            df["seq_subword_item_desc"] = parallelize_df_func(df["seq_item_desc"], df_get_subword)
            print("[%.5f] Done df_get_subword" % (time.time() - start_time))
        if EXTRACTED_SUBWORD_LIST:
            df["seq_subword_list_item_desc"] = parallelize_df_func(df["seq_item_desc"], df_get_subword_list)
            print("[%.5f] Done df_get_subword_list" % (time.time() - start_time))
    else:
        if EXTRACTED_BIGRAM:
            df["seq_bigram_item_desc"] = df_get_bigram(df["seq_item_desc"])
            print("[%.5f] Done df_get_bigram" % (time.time() - start_time))
        if EXTRACTED_TRIGRAM:
            df["seq_trigram_item_desc"] = df_get_trigram(df["seq_item_desc"])
            print("[%.5f] Done df_get_trigram" % (time.time() - start_time))
        if EXTRACTED_SUBWORD:
            df["seq_subword_item_desc"] = df_get_subword(df["seq_item_desc"])
            print("[%.5f] Done df_get_subword" % (time.time() - start_time))
        if EXTRACTED_SUBWORD_LIST:
            df["seq_subword_list_item_desc"] = df_get_subword_list(df["seq_item_desc"])
            print("[%.5f] Done df_get_subword_list" % (time.time() - start_time))
    if not VOCAB_HASHING_TRICK:
        if word_index is None:
            ##### word_index
            words = df.seq_name.tolist() + \
                    df.seq_item_desc.tolist()
                    # df.seq_category_name.tolist()
            word_index = get_word_index(words, MAX_NUM_WORDS, "word")
            del words
            gc.collect()
        if EXTRACTED_BIGRAM:
            if bigram_index is None:
                bigrams = df.seq_bigram_item_desc.tolist()
                bigram_index = get_word_index(bigrams, MAX_NUM_BIGRAMS, "bigram")
                del bigrams
                gc.collect()
        if EXTRACTED_TRIGRAM:
            if trigram_index is None:
                trigrams = df.seq_trigram_item_desc.tolist()
                trigram_index = get_word_index(trigrams, MAX_NUM_TRIGRAMS, "trigram")
                del trigrams
                gc.collect()
        if EXTRACTED_SUBWORD:
            if subword_index is None:
                subwords = df.seq_subword_item_desc.tolist()
                subword_index = get_word_index(subwords, MAX_NUM_SUBWORDS, "subword")
                del subwords
                gc.collect()
        if EXTRACTED_SUBWORD_LIST:
            if subword_list_index is None:
                subwords_list = df.seq_subword_list_item_desc.tolist()
                subwords_list = [w for sw in subwords_list for w in sw]
                subword_list_index = get_word_index(subwords_list, MAX_NUM_SUBWORDS_LIST, "subword_list")
                del subwords_list
                gc.collect()
        print("[%.5f] Done building vocab" % (time.time() - start_time))

        # faster
        # v = range(10000)
        # k = [str(i) for i in v]
        # vocab = dict(zip(k, v))
        # %timeit word2ind(word_lst, vocab)
        # 4.06 µs ± 63.8 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        def word2ind0(word_lst, vocab):
            vect = []
            for w in word_lst:
                if w in vocab:
                    vect.append(vocab[w])
            return vect

        # 4.46 µs ± 77.9 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        def word2ind1(word_lst, vocab):
            vect = [vocab[w] for w in word_lst if w in vocab]
            return vect

        # 13.3 µs ± 99.7 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        def word2ind2(word_lst, vocab):
            vect = []
            for w in word_lst:
                i = vocab.get(w)
                if i is not None:
                    vect.append(i)
            return vect

        # 14.6 µs ± 114 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        def word2ind3(word_lst, vocab):
            return [vocab.get(w, 1) for w in word_lst]

        word2ind = word2ind0

        def wordlist2ind0(word_list_lst, vocab):
            if len(word_list_lst) == 0:
                vect = [[]]
            else:
                vect = []
                for word_list in word_list_lst:
                    vect_ = []
                    for w in word_list:
                        if w in vocab:
                            vect_.append(vocab[w])
                    vect.append(vect_)
            return vect

        wordlist2ind = wordlist2ind0

        def word_lst_to_sequences(word_lst):
            return word2ind(word_lst, word_index)

        def df_word_lst_to_sequences(df):
            return df.apply(word_lst_to_sequences)

        def bigram_lst_to_sequences(word_lst):
            return word2ind(word_lst, bigram_index)

        def df_bigram_lst_to_sequences(df):
            return df.apply(bigram_lst_to_sequences)

        def trigram_lst_to_sequences(word_lst):
            return word2ind(word_lst, trigram_index)

        def df_trigram_lst_to_sequences(df):
            return df.apply(trigram_lst_to_sequences)

        def subword_lst_to_sequences(word_lst):
            return word2ind(word_lst, subword_index)

        def df_subword_lst_to_sequences(df):
            return df.apply(subword_lst_to_sequences)

        def subword_list_lst_to_sequences(word_list_lst):
            return wordlist2ind(word_list_lst, subword_list_index)

        def df_subword_list_lst_to_sequences(df):
            return df.apply(subword_list_lst_to_sequences)

        # print("   Transforming text to seq...")
        if USE_MULTITHREAD:
            df["seq_name"] = parallelize_df_func(df["seq_name"], df_word_lst_to_sequences)
            df["seq_item_desc"] = parallelize_df_func(df["seq_item_desc"], df_word_lst_to_sequences)
            # df["seq_category_name"] = parallelize_df_func(df["seq_category_name"], df_word_lst_to_sequences)
            print("[%.5f] Done df_word_lst_to_sequences" % (time.time() - start_time))
            if EXTRACTED_BIGRAM:
                df["seq_bigram_item_desc"] = parallelize_df_func(df["seq_bigram_item_desc"],
                                                                 df_bigram_lst_to_sequences)
                print("[%.5f] Done df_bigram_lst_to_sequences" % (time.time() - start_time))
            if EXTRACTED_TRIGRAM:
                df["seq_trigram_item_desc"] = parallelize_df_func(df["seq_trigram_item_desc"],
                                                                  df_trigram_lst_to_sequences)
                print("[%.5f] Done df_trigram_lst_to_sequences" % (time.time() - start_time))
            if EXTRACTED_SUBWORD:
                df["seq_subword_item_desc"] = parallelize_df_func(df["seq_subword_item_desc"],
                                                                  df_subword_lst_to_sequences)
                print("[%.5f] Done df_subword_lst_to_sequences" % (time.time() - start_time))
            if EXTRACTED_SUBWORD_LIST:
                df["seq_subword_list_item_desc"] = parallelize_df_func(df["seq_subword_list_item_desc"],
                                                                       df_subword_list_lst_to_sequences)
                print("[%.5f] Done df_subword_list_lst_to_sequences" % (time.time() - start_time))
        else:
            df["seq_name"] = df_word_lst_to_sequences(df["seq_name"])
            df["seq_item_desc"] = df_word_lst_to_sequences(df["seq_item_desc"])
            # df["seq_category_name"] = df_word_lst_to_sequences(df["seq_category_name"])
            print("[%.5f] Done df_word_lst_to_sequences" % (time.time() - start_time))
            if EXTRACTED_BIGRAM:
                df["seq_bigram_item_desc"] = df_bigram_lst_to_sequences(df["seq_bigram_item_desc"])
                print("[%.5f] Done df_bigram_lst_to_sequences" % (time.time() - start_time))
            if EXTRACTED_TRIGRAM:
                df["seq_trigram_item_desc"] = df_trigram_lst_to_sequences(df["seq_trigram_item_desc"])
                print("[%.5f] Done df_trigram_lst_to_sequences" % (time.time() - start_time))
            if EXTRACTED_SUBWORD:
                df["seq_subword_item_desc"] = df_subword_lst_to_sequences(df["seq_subword_item_desc"])
                print("[%.5f] Done df_subword_lst_to_sequences" % (time.time() - start_time))
            if EXTRACTED_SUBWORD_LIST:
                df["seq_subword_list_item_desc"] = df_subword_list_lst_to_sequences(df["seq_subword_list_item_desc"])
                print("[%.5f] Done df_subword_list_lst_to_sequences" % (time.time() - start_time))

    else:
        def word_lst_to_sequences_hash(word_lst):
            vect = [hashing_trick(w, MAX_NUM_WORDS) for w in word_lst]
            return vect

        def df_word_lst_to_sequences_hash(df):
            return df.apply(word_lst_to_sequences_hash)

        def bigram_lst_to_sequences_hash(word_lst):
            vect = [hashing_trick(w, MAX_NUM_BIGRAMS) for w in word_lst]
            return vect

        def df_bigram_lst_to_sequences_hash(df):
            return df.apply(bigram_lst_to_sequences_hash)

        def trigram_lst_to_sequences_hash(word_lst):
            vect = [hashing_trick(w, MAX_NUM_TRIGRAMS) for w in word_lst]
            return vect

        def df_trigram_lst_to_sequences_hash(df):
            return df.apply(trigram_lst_to_sequences_hash)

        def subword_lst_to_sequences_hash(word_lst):
            vect = [hashing_trick(w, MAX_NUM_SUBWORDS) for w in word_lst]
            return vect

        def df_subword_lst_to_sequences_hash(df):
            return df.apply(subword_lst_to_sequences_hash)

        def subword_list_lst_to_sequences_hash(word_list_lst):
            if len(word_list_lst) == 0:
                return [[]]
            else:
                vect = []
                for word_lst in word_list_lst:
                    vect.append([hashing_trick(w, MAX_NUM_SUBWORDS_LIST) for w in word_lst])
                return vect

        def df_subword_list_lst_to_sequences_hash(df):
            return df.apply(subword_list_lst_to_sequences_hash)

        # print("   Transforming text to seq...")
        if USE_MULTITHREAD:
            df["seq_name"] = parallelize_df_func(df["seq_name"], df_word_lst_to_sequences_hash)
            df["seq_item_desc"] = parallelize_df_func(df["seq_item_desc"], df_word_lst_to_sequences_hash)
            # df["seq_category_name"] = parallelize_df_func(df["seq_category_name"], df_word_lst_to_sequences_hash)
            gc.collect()
            print("[%.5f] Done df_word_lst_to_sequences_hash" % (time.time() - start_time))
            if EXTRACTED_BIGRAM:
                df["seq_bigram_item_desc"] = parallelize_df_func(df["seq_bigram_item_desc"],
                                                                 df_bigram_lst_to_sequences_hash)
                print("[%.5f] Done df_bigram_lst_to_sequences_hash" % (time.time() - start_time))
            if EXTRACTED_TRIGRAM:
                df["seq_trigram_item_desc"] = parallelize_df_func(df["seq_trigram_item_desc"],
                                                                  df_trigram_lst_to_sequences_hash)
                print("[%.5f] Done df_trigram_lst_to_sequences_hash" % (time.time() - start_time))
            if EXTRACTED_SUBWORD:
                df["seq_subword_item_desc"] = parallelize_df_func(df["seq_subword_item_desc"],
                                                                  df_subword_lst_to_sequences_hash)
                print("[%.5f] Done df_subword_lst_to_sequences_hash" % (time.time() - start_time))
            if EXTRACTED_SUBWORD_LIST:
                df["seq_subword_list_item_desc"] = parallelize_df_func(df["seq_subword_list_item_desc"],
                                                                       df_subword_list_lst_to_sequences_hash)
                print("[%.5f] Done df_subword_list_lst_to_sequences_hash" % (time.time() - start_time))
        else:
            df["seq_name"] = df_word_lst_to_sequences_hash(df["seq_name"])
            df["seq_item_desc"] = df_word_lst_to_sequences_hash(df["seq_item_desc"])
            # df["seq_category_name"] = df_word_lst_to_sequences_hash(df["seq_category_name"])
            gc.collect()
            print("[%.5f] Done df_word_lst_to_sequences_hash" % (time.time() - start_time))
            if EXTRACTED_BIGRAM:
                df["seq_bigram_item_desc"] = df_bigram_lst_to_sequences_hash(df["seq_bigram_item_desc"])
                print("[%.5f] Done df_bigram_lst_to_sequences_hash" % (time.time() - start_time))
            if EXTRACTED_TRIGRAM:
                df["seq_trigram_item_desc"] = df_trigram_lst_to_sequences_hash(df["seq_trigram_item_desc"])
                print("[%.5f] Done df_trigram_lst_to_sequences_hash" % (time.time() - start_time))
            if EXTRACTED_SUBWORD:
                df["seq_subword_item_desc"] = df_subword_lst_to_sequences_hash(df["seq_subword_item_desc"])
                print("[%.5f] Done df_subword_lst_to_sequences_hash" % (time.time() - start_time))
            if EXTRACTED_SUBWORD_LIST:
                df["seq_subword_list_item_desc"] = df_subword_list_lst_to_sequences_hash(
                    df["seq_subword_list_item_desc"])
                print("[%.5f] Done df_subword_list_lst_to_sequences_hash" % (time.time() - start_time))

    print("[%.5f] Done tokenize data" % (time.time() - start_time))

    if RUNNING_MODE != "submission":
        print('Average name sequence length: {}'.format(df["seq_name"].apply(len).mean()))
        print('Average item_desc sequence length: {}'.format(df["seq_item_desc"].apply(len).mean()))
        # print('Average brand_name sequence length: {}'.format(df["seq_brand_name"].apply(len).mean()))
        # print('Average category_name sequence length: {}'.format(df["seq_category_name"].apply(len).mean()))
        if EXTRACTED_SUBWORD:
            print('Average item_desc subword sequence length: {}'.format(
                df["seq_subword_item_desc"].apply(len).mean()))

    #### convert categorical variables
    if label_encoder is None:
        label_encoder = {}
        label_encoder["brand_name"] = MyLabelEncoder()
        df["brand_name_cat"] = label_encoder["brand_name"].fit_transform(df["brand_name"])
        label_encoder["category_name"] = MyLabelEncoder()
        df["category_name_cat"] = label_encoder["category_name"].fit_transform(df["category_name"])
        df.drop("brand_name", axis=1, inplace=True)
        df.drop("category_name", axis=1, inplace=True)
        gc.collect()
        for i in range(MAX_CATEGORY_NAME_LEN):
            label_encoder["category_name%d" % (i + 1)] = MyLabelEncoder()
            df["category_name%d_cat" % (i + 1)] = label_encoder["category_name%d" % (i + 1)].fit_transform(
                df["category_name%d" % (i + 1)])
            df.drop("category_name%d" % (i + 1), axis=1, inplace=True)
    else:
        df["brand_name_cat"] = label_encoder["brand_name"].transform(df["brand_name"])
        df["category_name_cat"] = label_encoder["category_name"].transform(df["category_name"])
        df.drop("brand_name", axis=1, inplace=True)
        df.drop("category_name", axis=1, inplace=True)
        gc.collect()
        for i in range(MAX_CATEGORY_NAME_LEN):
            df["category_name%d_cat" % (i + 1)] = label_encoder["category_name%d" % (i + 1)].transform(
                df["category_name%d" % (i + 1)])
            df.drop("category_name%d" % (i + 1), axis=1, inplace=True)
    print("[%.5f] Done Handling categorical variables" % (time.time() - start_time))


    if DUMP_DATA and RUNNING_MODE != "submission":
        try:
            with open(pkl_file, "wb") as f:
                pkl.dump(df, f)
        except:
            pass

    return df, word_index, bigram_index, trigram_index, subword_index, subword_list_index, label_encoder


feat_cols = [
    "missing_brand_name", "missing_category_name", "missing_item_desc",
]
NUM_VARS_DIM = len(feat_cols)


##########################################################################################
# KERAS MODEL DEFINITION AND TRAINING
##########################################################################################
def rmse(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    return np.sqrt(mean_squared_error(y_true, y_pred))


"""
modified from
https://github.com/guillaumegenthial/sequence_tagging
"""


def safe_len(x):
    try:
        return len(x)
    except:
        print(x)
        return 0


def get_max_seq_len(sequences, nlevels=1):
    max_seq_len = 1
    if nlevels == 1:
        for seq in sequences:
            max_seq_len = max(safe_len(seq), max_seq_len)
    elif nlevels == 2:
        for seqs in sequences:
            for seq in seqs:
                max_seq_len = max(safe_len(seq), max_seq_len)
    return max_seq_len


def _pad_sequences(sequences, max_length, pad_tok=0):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []
    sequence_padded_extend = sequence_padded.extend
    sequence_length_extend = sequence_length.extend
    for seq in sequences:
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - safe_len(seq), 0)
        sequence_padded_extend([seq_])
        sequence_length_extend([min(safe_len(seq), max_length)])
    return sequence_padded, sequence_length


def pad_sequences(sequences, max_seq_len=None, max_word_len=None, nlevels=1, pad_tok=0):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        if max_seq_len is None:
            max_seq_len = get_max_seq_len(sequences)
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                                          max_seq_len, pad_tok)
        sequence_padded = np.array(sequence_padded, dtype='int32')
        sequence_length = np.array(sequence_length, dtype='int32')
        return sequence_padded, sequence_length, max_seq_len

    elif nlevels == 2:
        if max_word_len is None:
            max_word_len = get_max_seq_len(sequences, nlevels=2)
        sequence_padded, sequence_length = [], []
        sequence_padded_extend = sequence_padded.extend
        sequence_length_extend = sequence_length.extend
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, max_word_len, pad_tok)
            sequence_padded_extend([sp])
            sequence_length_extend([sl])
        if max_seq_len is None:
            max_seq_len = get_max_seq_len(sequences)
        sequence_padded, _ = _pad_sequences(sequence_padded,
                                            max_seq_len, [pad_tok] * max_word_len)
        sequence_length, _ = _pad_sequences(sequence_length,
                                            max_seq_len, 0)
        sequence_padded = np.array(sequence_padded, dtype='int32')
        sequence_length = np.array(sequence_length, dtype='int32')
        return sequence_padded, sequence_length, max_seq_len, max_word_len


def get_tf_data(dataset, lbs, params):
    start_time = time.time()
    global MAX_NUM_BRANDS
    global MAX_NUM_CATEGORIES
    global MAX_NUM_CATEGORIES_LST
    global MAX_NUM_CONDITIONS

    if lbs is None:
        lbs = []
        lb = LabelBinarizer(sparse_output=True)
        item_condition_array = lb.fit_transform(dataset.item_condition_id).toarray()
        lbs.append(lb)

    else:
        lb = lbs[0]
        item_condition_array = lb.transform(dataset.item_condition_id).toarray()


    num_vars = dataset[feat_cols].values

    X = {}

    X['seq_name'] = keras_pad_sequences(dataset.seq_name, maxlen=params["max_sequence_length_name"],
                                        padding=params["pad_sequences_padding"],
                                        truncating=params["pad_sequences_truncating"])
    X["sequence_length_name"] = params["max_sequence_length_name"] * np.ones(dataset.shape[0])

    X['seq_item_desc'] = keras_pad_sequences(dataset.seq_item_desc, maxlen=params["max_sequence_length_item_desc"],
                                             padding=params["pad_sequences_padding"],
                                             truncating=params["pad_sequences_truncating"])
    X["sequence_length_item_desc"] = params["max_sequence_length_item_desc"] * np.ones(dataset.shape[0])

    X['seq_bigram_item_desc'] = keras_pad_sequences(dataset.seq_bigram_item_desc,
                                                    maxlen=params["max_sequence_length_item_desc"],
                                                    padding=params["pad_sequences_padding"],
                                                    truncating=params["pad_sequences_truncating"]) if params[
        "use_bigram"] else None

    X['seq_trigram_item_desc'] = keras_pad_sequences(dataset.seq_trigram_item_desc,
                                                     maxlen=params["max_sequence_length_item_desc"],
                                                     padding=params["pad_sequences_padding"],
                                                     truncating=params["pad_sequences_truncating"]) if params[
        "use_trigram"] else None

    X['seq_subword_item_desc'] = keras_pad_sequences(dataset.seq_subword_item_desc,
                                                     maxlen=params["max_sequence_length_item_desc_subword"],
                                                     padding=params["pad_sequences_padding"],
                                                     truncating=params["pad_sequences_truncating"]) if params[
        "use_subword"] else None
    X["sequence_length_item_desc_subword"] = params["max_sequence_length_item_desc_subword"] * np.ones(dataset.shape[0])

    X.update({
        'brand_name': dataset.brand_name_cat.values.reshape((-1, 1)),
        # 'category_name': dataset.category_name_cat.values.reshape((-1, 1)),
        'category_name1': dataset.category_name1_cat.values.reshape((-1, 1)),
        'category_name2': dataset.category_name2_cat.values.reshape((-1, 1)),
        'category_name3': dataset.category_name3_cat.values.reshape((-1, 1)),
        'item_condition_id': dataset.item_condition_id.values.reshape((-1, 1)),
        'item_condition': item_condition_array,
        'num_vars': num_vars,
        'shipping': dataset.shipping.values.reshape((-1, 1)),

    })

    print("[%.5f] Done get_tf_data." % (time.time() - start_time))
    return X, lbs, params


##################
# Model
##################
"""
https://explosion.ai/blog/deep-learning-formula-nlp
embed -> encode -> attend -> predict
"""
def batch_normalization(x, training, name):
    bn_train = tf.layers.batch_normalization(x, training=True, reuse=None, name=name)
    bn_inference = tf.layers.batch_normalization(x, training=False, reuse=True, name=name)
    z = tf.cond(training, lambda: bn_train, lambda: bn_inference)
    return z


#### Step 1
def embed(x, size, dim, seed=0, flatten=False, reduce_sum=False):
    # std = np.sqrt(2 / dim)
    std = 0.001
    minval = -std
    maxval = std
    emb = tf.Variable(tf.random_uniform([size, dim], minval, maxval, dtype=tf.float32, seed=seed))
    # None * max_seq_len * embed_dim
    out = tf.nn.embedding_lookup(emb, x)
    if flatten:
        out = tf.layers.flatten(out)
    if reduce_sum:
        out = tf.reduce_sum(out, axis=1)
    return out


def embed_subword(x, size, dim, sequence_length, seed=0, mask_zero=False, maxlen=None):
    # std = np.sqrt(2 / dim)
    std = 0.001
    minval = -std
    maxval = std
    emb = tf.Variable(tf.random_uniform([size, dim], minval, maxval, dtype=tf.float32, seed=seed))
    # None * max_seq_len * max_word_len * embed_dim
    out = tf.nn.embedding_lookup(emb, x)
    if mask_zero:
        # word_len: None * max_seq_len
        # mask: shape=None * max_seq_len * max_word_len
        mask = tf.sequence_mask(sequence_length, maxlen)
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.cast(mask, tf.float32)
        out = out * mask
    # None * max_seq_len * embed_dim
    # according to facebook subword paper, it's sum
    out = tf.reduce_sum(out, axis=2)
    return out


def word_dropout(x, training, dropout=0, seed=0):
    # word dropout (dropout the entire embedding for some words)
    """
    tf.layers.Dropout doesn't work as it can't switch training or inference
    """
    if dropout > 0:
        input_shape = tf.shape(x)
        noise_shape = [input_shape[0], input_shape[1], 1]
        x = tf.layers.Dropout(rate=dropout, noise_shape=noise_shape, seed=seed)(x, training=training)
    return x


#### Step 2
def fasttext(x):
    return x


def timedistributed_conv1d(x, filter_size):
    """not working"""
    # None * embed_dim * step_dim
    input_shape = tf.shape(x)
    step_dim = input_shape[1]
    embed_dim = input_shape[2]
    x = tf.transpose(x, [0, 2, 1])
    # None * embed_dim * step_dim
    x = tf.reshape(x, [input_shape[0] * embed_dim, step_dim, 1])
    conv = tf.layers.Conv1D(
        filters=1,
        kernel_size=filter_size,
        padding="same",
        activation=None,
        strides=1)(x)
    conv = tf.reshape(conv, [input_shape[0], embed_dim, step_dim])
    conv = tf.transpose(conv, [0, 2, 1])
    return conv


def textcnn(x, num_filters=8, filter_sizes=[2, 3], timedistributed=False):
    # x: None * step_dim * embed_dim
    conv_blocks = []
    for i, filter_size in enumerate(filter_sizes):
        if timedistributed:
            conv = timedistributed_conv1d(x, filter_size)
        else:
            conv = tf.layers.Conv1D(
                filters=num_filters,
                kernel_size=filter_size,
                padding="same",
                activation=None,
                strides=1)(x)
        conv = tf.layers.BatchNormalization()(conv)
        conv = tf.nn.relu(conv)
        conv_blocks.append(conv)
    if len(conv_blocks) > 1:
        z = tf.concat(conv_blocks, axis=-1)
    else:
        z = conv_blocks[0]
    return z


def textrnn(x, num_units, cell_type, sequence_length, mask_zero=False, scope=None):
    if cell_type == "gru":
        cell_fw = tf.nn.rnn_cell.GRUCell(num_units)
    elif cell_type == "lstm":
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_units)
    if mask_zero:
        x, _ = tf.nn.dynamic_rnn(cell_fw, x, dtype=tf.float32, sequence_length=sequence_length, scope=scope)
    else:
        x, _ = tf.nn.dynamic_rnn(cell_fw, x, dtype=tf.float32, sequence_length=None, scope=scope)
    return x


def textbirnn(x, num_units, cell_type, sequence_length, mask_zero=False, scope=None):
    if cell_type == "gru":
        cell_fw = tf.nn.rnn_cell.GRUCell(num_units)
        cell_bw = tf.nn.rnn_cell.GRUCell(num_units)
    elif cell_type == "lstm":
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_units)
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_units)
    if mask_zero:
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, x, dtype=tf.float32, sequence_length=sequence_length, scope=scope)
    else:
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, x, dtype=tf.float32, sequence_length=None, scope=scope)
    x = 0.5 * (output_fw + output_bw)
    return x


def encode(x, method, params, sequence_length, mask_zero=False, scope=None):
    """
    :param x: shape=(None,seqlen,dim)
    :param params:
    :return: shape=(None,seqlen,dim)
    """
    if method == "fasttext":
        z = fasttext(x)
    elif method == "textcnn":
        z = textcnn(x, num_filters=params["cnn_num_filters"], filter_sizes=params["cnn_filter_sizes"],
                    timedistributed=params["cnn_timedistributed"])
    elif method == "textrnn":
        z = textrnn(x, num_units=params["rnn_num_units"], cell_type=params["rnn_cell_type"],
                    sequence_length=sequence_length, mask_zero=mask_zero, scope=scope)
    elif method == "textbirnn":
        z = textbirnn(x, num_units=params["rnn_num_units"], cell_type=params["rnn_cell_type"],
                      sequence_length=sequence_length, mask_zero=mask_zero, scope=scope)
    elif method == "fasttext+textcnn":
        z_f = fasttext(x)
        z_c = textcnn(x, num_filters=params["cnn_num_filters"], filter_sizes=params["cnn_filter_sizes"],
                      timedistributed=params["cnn_timedistributed"])
        z = tf.concat([z_f, z_c], axis=-1)
    elif method == "fasttext+textrnn":
        z_f = fasttext(x)
        z_r = textrnn(x, num_units=params["rnn_num_units"], cell_type=params["rnn_cell_type"],
                      sequence_length=sequence_length, mask_zero=mask_zero, scope=scope)
        z = tf.concat([z_f, z_r], axis=-1)
    elif method == "fasttext+textbirnn":
        z_f = fasttext(x)
        z_b = textbirnn(x, num_units=params["rnn_num_units"], cell_type=params["rnn_cell_type"],
                        sequence_length=sequence_length, mask_zero=mask_zero, scope=scope)
        z = tf.concat([z_f, z_b], axis=-1)
    elif method == "fasttext+textcnn+textrnn":
        z_f = fasttext(x)
        z_c = textcnn(x, num_filters=params["cnn_num_filters"], filter_sizes=params["cnn_filter_sizes"],
                      timedistributed=params["cnn_timedistributed"])
        z_r = textrnn(x, num_units=params["rnn_num_units"], cell_type=params["rnn_cell_type"],
                      sequence_length=sequence_length, mask_zero=mask_zero, scope=scope)
        z = tf.concat([z_f, z_c, z_r], axis=-1)
    elif method == "fasttext+textcnn+textbirnn":
        z_f = fasttext(x)
        z_c = textcnn(x, num_filters=params["cnn_num_filters"], filter_sizes=params["cnn_filter_sizes"],
                      timedistributed=params["cnn_timedistributed"])
        z_b = textbirnn(x, num_units=params["rnn_num_units"], cell_type=params["rnn_cell_type"],
                        sequence_length=sequence_length, mask_zero=mask_zero, scope=scope)
        z = tf.concat([z_f, z_c, z_b], axis=-1)
    return z


#### Step 3
def attention(x, feature_dim, sequence_length, mask_zero=False, maxlen=None, epsilon=1e-8, seed=0):
    input_shape = tf.shape(x)
    step_dim = input_shape[1]
    # feature_dim = input_shape[2]
    x = tf.reshape(x, [-1, feature_dim])
    """
    The last dimension of the inputs to `Dense` should be defined. Found `None`.

    cann't not use `tf.layers.Dense` here
    eij = tf.layers.Dense(1)(x)

    see: https://github.com/tensorflow/tensorflow/issues/13348
    workaround: specify the feature_dim as input
    """

    eij = tf.layers.Dense(1, activation=tf.nn.tanh, kernel_initializer=tf.glorot_uniform_initializer(seed=seed),
                          dtype=tf.float32, bias_initializer=tf.zeros_initializer())(x)
    eij = tf.reshape(eij, [-1, step_dim])
    a = tf.exp(eij)

    # apply mask after the exp. will be re-normalized next
    if mask_zero:
        # None * step_dim
        mask = tf.sequence_mask(sequence_length, maxlen)
        mask = tf.cast(mask, tf.float32)
        a = a * mask

    # in some cases especially in the early stages of training the sum may be almost zero
    a /= tf.cast(tf.reduce_sum(a, axis=1, keep_dims=True) + epsilon, tf.float32)

    a = tf.expand_dims(a, axis=-1)
    return a


def attend(x, sequence_length=None, method="ave", context=None, feature_dim=None, mask_zero=False, maxlen=None,
           epsilon=1e-8, bn=True, training=False, seed=0, reuse=True, name="attend"):
    if method == "ave":
        if mask_zero:
            # None * step_dim
            mask = tf.sequence_mask(sequence_length, maxlen)
            mask = tf.reshape(mask, (-1, tf.shape(x)[1], 1))
            mask = tf.cast(mask, tf.float32)
            z = tf.reduce_sum(x * mask, axis=1)
            l = tf.reduce_sum(mask, axis=1)
            # in some cases especially in the early stages of training the sum may be almost zero
            z /= tf.cast(l + epsilon, tf.float32)
        else:
            z = tf.reduce_mean(x, axis=1)
    elif method == "sum":
        if mask_zero:
            # None * step_dim
            mask = tf.sequence_mask(sequence_length, maxlen)
            mask = tf.reshape(mask, (-1, tf.shape(x)[1], 1))
            mask = tf.cast(mask, tf.float32)
            z = tf.reduce_sum(x * mask, axis=1)
        else:
            z = tf.reduce_sum(x, axis=1)
    elif method == "max":
        if mask_zero:
            # None * step_dim
            mask = tf.sequence_mask(sequence_length, maxlen)
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.tile(mask, (1, 1, tf.shape(x)[2]))
            masked_data = tf.where(tf.equal(mask, tf.zeros_like(mask)),
                                   tf.ones_like(x) * -np.inf, x)  # if masked assume value is -inf
            z = tf.reduce_max(masked_data, axis=1)
        else:
            z = tf.reduce_max(x, axis=1)
    elif method == "attention":
        if context is not None:
            step_dim = tf.shape(x)[1]
            context = tf.expand_dims(context, axis=1)
            context = tf.tile(context, [1, step_dim, 1])
            y = tf.concat([x, context], axis=-1)
        else:
            y = x
        a = attention(y, feature_dim, sequence_length, mask_zero, maxlen, seed=seed)
        z = tf.reduce_sum(x * a, axis=1)
    if bn:
        # training=False has slightly better performance
        z = tf.layers.BatchNormalization()(z, training=False)
        # z = batch_normalization(z, training=training, name=name)
    return z


#### Step 4
def _dense_block_mode1(x, hidden_units, dropouts, densenet=False, training=False, seed=0, bn=False, name="dense_block"):
    """
    :param x:
    :param hidden_units:
    :param dropouts:
    :param densenet: enable densenet
    :return:
    Ref: https://github.com/titu1994/DenseNet
    """
    for i, (h, d) in enumerate(zip(hidden_units, dropouts)):
        z = tf.layers.Dense(h, kernel_initializer=tf.glorot_uniform_initializer(seed=seed * i),
                            dtype=tf.float32,
                            bias_initializer=tf.zeros_initializer())(x)
        if bn:
            z = batch_normalization(z, training=training, name=name+"-"+str(i))
        z = tf.nn.relu(z)
        # z = tf.nn.selu(z)
        z = tf.layers.Dropout(d, seed=seed * i)(z, training=training) if d > 0 else z
        if densenet:
            x = tf.concat([x, z], axis=-1)
        else:
            x = z
    return x


def _dense_block_mode2(x, hidden_units, dropouts, densenet=False, training=False, seed=0, bn=False, name="dense_block"):
    """
    :param x:
    :param hidden_units:
    :param dropouts:
    :param densenet: enable densenet
    :return:
    Ref: https://github.com/titu1994/DenseNet
    """
    for i, (h, d) in enumerate(zip(hidden_units, dropouts)):
        if bn:
            z = batch_normalization(x, training=training, name=name + "-" + str(i))
        z = tf.nn.relu(z)
        z = tf.layers.Dropout(d, seed=seed * i)(z, training=training) if d > 0 else z
        z = tf.layers.Dense(h, kernel_initializer=tf.glorot_uniform_initializer(seed=seed * i), dtype=tf.float32,
                            bias_initializer=tf.zeros_initializer())(z)
        if densenet:
            x = tf.concat([x, z], axis=-1)
        else:
            x = z
    return x


def dense_block(x, hidden_units, dropouts, densenet=False, training=False, seed=0, bn=False, name="dense_block"):
    return _dense_block_mode1(x, hidden_units, dropouts, densenet, training, seed, bn, name)


def _resnet_branch_mode1(x, hidden_units, dropouts, training, seed=0):
    h1, h2, h3 = hidden_units
    dr1, dr2, dr3 = dropouts
    # branch 2
    x2 = tf.layers.Dense(h1, kernel_initializer=tf.glorot_uniform_initializer(seed=seed * 2), dtype=tf.float32,
                         bias_initializer=tf.zeros_initializer())(x)
    x2 = tf.layers.BatchNormalization()(x2)
    x2 = tf.nn.relu(x2)
    x2 = tf.layers.Dropout(dr1, seed=seed * 1)(x2, training=training) if dr1 > 0 else x2

    x2 = tf.layers.Dense(h2, kernel_initializer=tf.glorot_uniform_initializer(seed=seed * 3), dtype=tf.float32,
                         bias_initializer=tf.zeros_initializer())(x2)
    x2 = tf.layers.BatchNormalization()(x2)
    x2 = tf.nn.relu(x2)
    x2 = tf.layers.Dropout(dr2, seed=seed * 2)(x2, training=training) if dr2 > 0 else x2

    x2 = tf.layers.Dense(h3, kernel_initializer=tf.glorot_uniform_initializer(seed=seed * 4), dtype=tf.float32,
                         bias_initializer=tf.zeros_initializer())(x2)
    x2 = tf.layers.BatchNormalization()(x2)

    return x2


def _resnet_block_mode1(x, hidden_units, dropouts, cardinality=1, dense_shortcut=False, training=False, seed=0):
    """A block that has a dense layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    h1, h2, h3 = hidden_units
    dr1, dr2, dr3 = dropouts

    xs = []
    # branch 0
    if dense_shortcut:
        x0 = tf.layers.Dense(h3, kernel_initializer=tf.glorot_uniform_initializer(seed=seed * 1), dtype=tf.float32,
                             bias_initializer=tf.zeros_initializer())(x)
        x0 = tf.layers.BatchNormalization()(x0)
        xs.append(x0)
    else:
        xs.append(x)

    # branch 1 ~ cardinality
    for i in range(cardinality):
        xs.append(_resnet_branch_mode1(x, hidden_units, dropouts, training, seed))

    x = tf.add_n(xs)
    x = tf.nn.relu(x)
    x = tf.layers.Dropout(dr3, seed=seed * 4)(x, training=training) if dr3 > 0 else x
    return x


def _resnet_branch_mode2(x, hidden_units, dropouts, training=False, seed=0):
    h1, h2, h3 = hidden_units
    dr1, dr2, dr3 = dropouts
    # branch 2: bn-relu->weight
    x2 = tf.layers.BatchNormalization()(x)
    x2 = tf.nn.relu(x2)
    x2 = tf.layers.Dropout(dr1)(x2, training=training) if dr1 > 0 else x2
    x2 = tf.layers.Dense(h1, kernel_initializer=tf.glorot_uniform_initializer(seed * 1), dtype=tf.float32,
                         bias_initializer=tf.zeros_initializer())(x2)

    x2 = tf.layers.BatchNormalization()(x2)
    x2 = tf.nn.relu(x2)
    x2 = tf.layers.Dropout(dr2)(x2, training=training) if dr2 > 0 else x2
    x2 = tf.layers.Dense(h2, kernel_initializer=tf.glorot_uniform_initializer(seed * 2), dtype=tf.float32,
                         bias_initializer=tf.zeros_initializer())(x2)

    x2 = tf.layers.BatchNormalization()(x2)
    x2 = tf.nn.relu(x2)
    x2 = tf.layers.Dropout(dr3)(x2, training=training) if dr3 > 0 else x2
    x2 = tf.layers.Dense(h3, kernel_initializer=tf.glorot_uniform_initializer(seed * 3), dtype=tf.float32,
                         bias_initializer=tf.zeros_initializer())(x2)

    return x2


def _resnet_block_mode2(x, hidden_units, dropouts, cardinality=1, dense_shortcut=False, training=False, seed=0):
    """A block that has a dense layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    h1, h2, h3 = hidden_units
    dr1, dr2, dr3 = dropouts

    xs = []
    # branch 0
    if dense_shortcut:
        x0 = tf.layers.Dense(h3, kernel_initializer=tf.glorot_uniform_initializer(seed * 1), dtype=tf.float32,
                             bias_initializer=tf.zeros_initializer())(x)
        xs.append(x0)
    else:
        xs.append(x)

    # branch 1 ~ cardinality
    for i in range(cardinality):
        xs.append(_resnet_branch_mode2(x, hidden_units, dropouts, training, seed))

    x = tf.add_n(xs)
    return x


def resnet_block(input_tensor, hidden_units, dropouts, cardinality=1, dense_shortcut=False, training=False, seed=0):
    return _resnet_block_mode2(input_tensor, hidden_units, dropouts, cardinality, dense_shortcut, training, seed)


#### model
class MercariNet(BaseEstimator, TransformerMixin):
    def __init__(self, params):
        self.params = params
        _makedirs(self.params["model_dir"], force=True)
        self._init_graph()
        self.gvars_state_list = []

        # 14
        self.bias = 0.01228477
        self.weights = [
            0.00599607, 0.02999416, 0.05985384, 0.20137787, 0.03178938, 0.04612812,
            0.05384821, 0.10121514, 0.05915169, 0.05521121, 0.06448063, 0.0944233,
            0.08306157, 0.11769992
        ]
        self.weights = np.array(self.weights).reshape(-1, 1)

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.params["random_seed"])

            #### input
            self.training = tf.placeholder(tf.bool, shape=[], name="training")
            # seq
            # None * max_sequence_length_name
            self.seq_name = tf.placeholder(tf.int32, shape=[None, None], name="seq_name")
            # None * max_sequence_length_item_desc
            self.seq_item_desc = tf.placeholder(tf.int32, shape=[None, None], name="seq_item_desc")
            # # None * max_sequence_length_category_name
            self.seq_category_name = tf.placeholder(tf.int32, shape=[None, None], name="seq_category_name")
            if self.params["use_bigram"]:
                # None * max_sequence_length_item_desc
                self.seq_bigram_item_desc = tf.placeholder(tf.int32, shape=[None, None], name="seq_bigram_item_desc")
            if self.params["use_trigram"]:
                # None * max_sequence_length_item_desc
                self.seq_trigram_item_desc = tf.placeholder(tf.int32, shape=[None, None], name="seq_trigram_item_desc")
            if self.params["use_subword"]:
                # None * max_sequence_length_item_desc
                self.seq_subword_item_desc = tf.placeholder(tf.int32, shape=[None, None], name="seq_subword_item_desc")
            if self.params["use_subword_list"]:
                # None * max_sequence_length_item_desc * max_word_len
                self.seq_subword_list_item_desc = tf.placeholder(tf.int32, shape=[None, None, None],
                                                                 name="seq_subword_list_item_desc")

            # placeholder for length
            self.sequence_length_name = tf.placeholder(tf.int32, shape=[None], name="sequence_length_name")
            self.sequence_length_item_desc = tf.placeholder(tf.int32, shape=[None], name="sequence_length_item_desc")
            self.sequence_length_category_name = tf.placeholder(tf.int32, shape=[None],
                                                                name="sequence_length_category_name")
            self.sequence_length_item_desc_subword = tf.placeholder(tf.int32, shape=[None],
                                                                    name="sequence_length_item_desc_subword")
            self.word_length = tf.placeholder(tf.int32, shape=[None, None], name="word_length")

            # other context
            self.brand_name = tf.placeholder(tf.int32, shape=[None, 1], name="brand_name")
            # self.category_name = tf.placeholder(tf.int32, shape=[None, 1], name="category_name")
            self.category_name1 = tf.placeholder(tf.int32, shape=[None, 1], name="category_name1")
            self.category_name2 = tf.placeholder(tf.int32, shape=[None, 1], name="category_name2")
            self.category_name3 = tf.placeholder(tf.int32, shape=[None, 1], name="category_name3")
            self.item_condition_id = tf.placeholder(tf.int32, shape=[None, 1], name="item_condition_id")
            self.item_condition = tf.placeholder(tf.float32, shape=[None, MAX_NUM_CONDITIONS], name="item_condition")
            self.shipping = tf.placeholder(tf.int32, shape=[None, 1], name="shipping")
            self.num_vars = tf.placeholder(tf.float32, shape=[None, NUM_VARS_DIM], name="num_vars")

            # target
            self.target = tf.placeholder(tf.float32, shape=[None, 1], name="target")

            #### embed
            # embed seq
            # None * max_sequence_length_name * embedding_dim
            # std = np.sqrt(2 / self.params["embedding_dim"])
            std = 0.001
            minval = -std
            maxval = std
            emb_word = tf.Variable(
                tf.random_uniform([MAX_NUM_WORDS + 1, self.params["embedding_dim"]], minval, maxval,
                                  seed=self.params["random_seed"],
                                  dtype=tf.float32))
            # emb_word2 = tf.Variable(tf.random_uniform([MAX_NUM_WORDS + 1, self.params["embedding_dim"]], minval, maxval,
            #                                     seed=self.params["random_seed"],
            #                                     dtype=tf.float32))
            emb_seq_name = tf.nn.embedding_lookup(emb_word, self.seq_name)
            if self.params["embedding_dropout"] > 0.:
                emb_seq_name = word_dropout(emb_seq_name, training=self.training,
                                            dropout=self.params["embedding_dropout"],
                                            seed=self.params["random_seed"])
            # None * max_sequence_length_item_desc * embedding_dim
            emb_seq_item_desc = tf.nn.embedding_lookup(emb_word, self.seq_item_desc)
            if self.params["embedding_dropout"] > 0.:
                emb_seq_item_desc = word_dropout(emb_seq_item_desc, training=self.training,
                                                 dropout=self.params["embedding_dropout"],
                                                 seed=self.params["random_seed"])
            # # None * max_sequence_length_category_name * embedding_dim
            # emb_seq_category_name = tf.nn.embedding_lookup(emb_word, self.seq_category_name)
            # if self.params["embedding_dropout"] > 0.:
            #     emb_seq_category_name = word_dropout(emb_seq_category_name, training=self.training,
            #                                      dropout=self.params["embedding_dropout"],
            #                                      seed=self.params["random_seed"])
            if self.params["use_bigram"]:
                # None * max_sequence_length_item_desc * embedding_dim
                emb_seq_bigram_item_desc = embed(self.seq_bigram_item_desc, MAX_NUM_BIGRAMS + 1,
                                                 self.params["embedding_dim"], seed=self.params["random_seed"])
                if self.params["embedding_dropout"] > 0.:
                    emb_seq_bigram_item_desc = word_dropout(emb_seq_bigram_item_desc, training=self.training,
                                                            dropout=self.params["embedding_dropout"],
                                                            seed=self.params["random_seed"])
            if self.params["use_trigram"]:
                # None * max_sequence_length_item_desc * embedding_dim
                emb_seq_trigram_item_desc = embed(self.seq_trigram_item_desc, MAX_NUM_TRIGRAMS + 1,
                                                  self.params["embedding_dim"], seed=self.params["random_seed"])
                if self.params["embedding_dropout"] > 0.:
                    emb_seq_trigram_item_desc = word_dropout(emb_seq_trigram_item_desc, training=self.training,
                                                             dropout=self.params["embedding_dropout"],
                                                             seed=self.params["random_seed"])
            if self.params["use_subword"]:
                # None * max_sequence_length_item_desc * embedding_dim
                emb_seq_subword_item_desc = embed(self.seq_subword_item_desc, MAX_NUM_SUBWORDS + 1,
                                                  self.params["embedding_dim"], seed=self.params["random_seed"])
                if self.params["embedding_dropout"] > 0.:
                    emb_seq_subword_item_desc = word_dropout(emb_seq_subword_item_desc, training=self.training,
                                                             dropout=self.params["embedding_dropout"],
                                                             seed=self.params["random_seed"])
            if self.params["use_subword_list"]:
                # None * max_sequence_length_item_desc * embedding_dim
                emb_seq_subword_list_item_desc = embed_subword(self.seq_subword_list_item_desc,
                                                               MAX_NUM_SUBWORDS_LIST + 1,
                                                               self.params["embedding_dim"],
                                                               mask_zero=self.params["embedding_mask_zero_subword"],
                                                               sequence_length=self.word_length,
                                                               maxlen=self.params["max_word_len"],
                                                               seed=self.params["random_seed"])
                if self.params["embedding_dropout"] > 0.:
                    emb_seq_subword_list_item_desc = word_dropout(emb_seq_subword_list_item_desc,
                                                                  training=self.training,
                                                                  dropout=self.params["embedding_dropout"],
                                                                  seed=self.params["random_seed"])
                emb_seq_item_desc = 0.5 * (emb_seq_item_desc + emb_seq_subword_list_item_desc)

            # embed other context
            std = 0.001
            minval = -std
            maxval = std
            emb_brand = tf.Variable(
                tf.random_uniform([MAX_NUM_BRANDS, self.params["embedding_dim"]], minval, maxval,
                                  seed=self.params["random_seed"],
                                  dtype=tf.float32))
            emb_brand_name = tf.nn.embedding_lookup(emb_brand, self.brand_name)
            # emb_brand_name = embed(self.brand_name, MAX_NUM_BRANDS, self.params["embedding_dim"],
            #                        flatten=False, seed=self.params["random_seed"])
            # emb_category_name = embed(self.category_name, MAX_NUM_CATEGORIES, self.params["embedding_dim"],
            #                           flatten=False)
            emb_category_name1 = embed(self.category_name1, MAX_NUM_CATEGORIES_LST[0], self.params["embedding_dim"],
                                       flatten=False, seed=self.params["random_seed"])
            emb_category_name2 = embed(self.category_name2, MAX_NUM_CATEGORIES_LST[1], self.params["embedding_dim"],
                                       flatten=False, seed=self.params["random_seed"])
            emb_category_name3 = embed(self.category_name3, MAX_NUM_CATEGORIES_LST[2], self.params["embedding_dim"],
                                       flatten=False, seed=self.params["random_seed"])
            emb_item_condition = embed(self.item_condition_id, MAX_NUM_CONDITIONS + 1, self.params["embedding_dim"],
                                       flatten=False, seed=self.params["random_seed"])
            emb_shipping = embed(self.shipping, MAX_NUM_SHIPPINGS, self.params["embedding_dim"],
                                 flatten=False, seed=self.params["random_seed"])

            #### encode
            enc_seq_name = encode(emb_seq_name, method=self.params["encode_method"],
                                  params=self.params,
                                  sequence_length=self.sequence_length_name,
                                  mask_zero=self.params["embedding_mask_zero"],
                                  scope="enc_seq_name")
            enc_seq_item_desc = encode(emb_seq_item_desc, method=self.params["encode_method"],
                                       params=self.params, sequence_length=self.sequence_length_item_desc,
                                       mask_zero=self.params["embedding_mask_zero"],
                                       scope="enc_seq_item_desc")
            # enc_seq_category_name = encode(emb_seq_category_name, method=self.params["encode_method"],
            #                                params=self.params, sequence_length=self.sequence_length_category_name,
            #                                mask_zero=self.params["embedding_mask_zero"],
            #                                scope="enc_seq_category_name")
            if self.params["use_bigram"]:
                enc_seq_bigram_item_desc = encode(emb_seq_bigram_item_desc, method="fasttext",
                                                  params=self.params,
                                                  sequence_length=self.sequence_length_item_desc,
                                                  mask_zero=self.params["embedding_mask_zero"],
                                                  scope="enc_seq_bigram_item_desc")
            if self.params["use_trigram"]:
                enc_seq_trigram_item_desc = encode(emb_seq_trigram_item_desc, method="fasttext",
                                                   params=self.params,
                                                   sequence_length=self.sequence_length_item_desc,
                                                   mask_zero=self.params["embedding_mask_zero"],
                                                   scope="enc_seq_trigram_item_desc")
            # use fasttext encode method for the following
            if self.params["use_subword"]:
                enc_seq_subword_item_desc = encode(emb_seq_subword_item_desc, method="fasttext",
                                                   params=self.params,
                                                   sequence_length=self.sequence_length_item_desc_subword,
                                                   mask_zero=self.params["embedding_mask_zero"],
                                                   scope="enc_seq_subword_item_desc")

            context = tf.concat([
                # att_seq_category_name,
                tf.layers.flatten(emb_brand_name),
                # tf.layers.flatten(emb_category_name),
                tf.layers.flatten(emb_category_name1),
                tf.layers.flatten(emb_category_name2),
                tf.layers.flatten(emb_category_name3),
                self.item_condition,
                tf.cast(self.shipping, tf.float32),
                self.num_vars],
                axis=-1, name="context")
            context_size = self.params["encode_text_dim"] * 0 + \
                           self.params["embedding_dim"] * 4 + \
                           self.params["item_condition_size"] + \
                           self.params["shipping_size"] + \
                           self.params["num_vars_size"]

            feature_dim = context_size + self.params["encode_text_dim"]
            # context = None
            feature_dim = self.params["encode_text_dim"]
            att_seq_name = attend(enc_seq_name, method=self.params["attend_method"],
                                  context=None, feature_dim=feature_dim,
                                  sequence_length=self.sequence_length_name,
                                  maxlen=self.params["max_sequence_length_name"],
                                  mask_zero=self.params["embedding_mask_zero"],
                                  training=self.training,
                                  seed=self.params["random_seed"],
                                  name="att_seq_name_attend")
            att_seq_item_desc = attend(enc_seq_item_desc, method=self.params["attend_method"],
                                       context=None, feature_dim=feature_dim,
                                       sequence_length=self.sequence_length_item_desc,
                                       maxlen=self.params["max_sequence_length_item_desc"],
                                       mask_zero=self.params["embedding_mask_zero"],
                                       training=self.training,
                                       seed=self.params["random_seed"],
                                       name="att_seq_item_desc_attend")
            if self.params["encode_text_dim"] != self.params["embedding_dim"]:
                att_seq_name = tf.layers.Dense(self.params["embedding_dim"])(att_seq_name)
                att_seq_item_desc = tf.layers.Dense(self.params["embedding_dim"])(att_seq_item_desc)
            # since the following use fasttext encode, the `encode_text_dim` is embedding_dim
            feature_dim = context_size + self.params["embedding_dim"]
            feature_dim = self.params["embedding_dim"]
            if self.params["use_bigram"]:
                att_seq_bigram_item_desc = attend(enc_seq_bigram_item_desc, method=self.params["attend_method"],
                                                  context=None, feature_dim=feature_dim,
                                                  sequence_length=self.sequence_length_item_desc,
                                                  maxlen=self.params["max_sequence_length_item_desc"],
                                                  mask_zero=self.params["embedding_mask_zero"],
                                                  training=self.training,
                                                  seed=self.params["random_seed"],
                                                  name="att_seq_bigram_item_desc_attend")
                # reshape
                if self.params["encode_text_dim"] != self.params["embedding_dim"]:
                    att_seq_bigram_item_desc = tf.layers.Dense(self.params["embedding_dim"],
                                                               kernel_initializer=tf.glorot_uniform_initializer(),
                                                               dtype=tf.float32, bias_initializer=tf.zeros_initializer())(att_seq_bigram_item_desc)
            if self.params["use_trigram"]:
                att_seq_trigram_item_desc = attend(enc_seq_trigram_item_desc, method=self.params["attend_method"],
                                                   context=None, feature_dim=feature_dim,
                                                   sequence_length=self.sequence_length_item_desc,
                                                   maxlen=self.params["max_sequence_length_item_desc"],
                                                   mask_zero=self.params["embedding_mask_zero"],
                                                   training=self.training,
                                                   seed=self.params["random_seed"],
                                                   name="att_seq_trigram_item_desc_attend")
                # reshape
                if self.params["encode_text_dim"] != self.params["embedding_dim"]:
                    att_seq_trigram_item_desc = tf.layers.Dense(self.params["embedding_dim"],
                                                                kernel_initializer=tf.glorot_uniform_initializer(),
                                                                dtype=tf.float32, bias_initializer=tf.zeros_initializer())(att_seq_trigram_item_desc)
            feature_dim = context_size + self.params["embedding_dim"]
            if self.params["use_subword"]:
                att_seq_subword_item_desc = attend(enc_seq_subword_item_desc, method="ave",
                                                   context=None, feature_dim=feature_dim,
                                                   sequence_length=self.sequence_length_item_desc_subword,
                                                   maxlen=self.params["max_sequence_length_item_desc_subword"],
                                                   mask_zero=self.params["embedding_mask_zero"],
                                                   training=self.training,
                                                   seed=self.params["random_seed"],
                                                   name="att_seq_subword_item_desc_attend")
                # reshape
                if self.params["encode_text_dim"] != self.params["embedding_dim"]:
                    att_seq_subword_item_desc = tf.layers.Dense(self.params["embedding_dim"],
                                                                kernel_initializer=tf.glorot_uniform_initializer(),
                                                                dtype=tf.float32, bias_initializer=tf.zeros_initializer())(att_seq_subword_item_desc)

            deep_list = []
            if self.params["enable_deep"]:
                # common
                common_list = [
                    # emb_seq_category_name,
                    emb_brand_name,
                    # emb_category_name,
                    emb_category_name1,
                    emb_category_name2,
                    emb_category_name3,
                    emb_item_condition,
                    emb_shipping

                ]
                tmp_common = tf.concat(common_list, axis=1)

                # word level fm for seq_name and others
                tmp_name = tf.concat([emb_seq_name, tmp_common], axis=1)
                sum_squared_name = tf.square(tf.reduce_sum(tmp_name, axis=1))
                squared_sum_name = tf.reduce_sum(tf.square(tmp_name), axis=1)
                fm_name = 0.5 * (sum_squared_name - squared_sum_name)

                # word level fm for seq_item_desc and others
                tmp_item_desc = tf.concat([emb_seq_item_desc, tmp_common], axis=1)
                sum_squared_item_desc = tf.square(tf.reduce_sum(tmp_item_desc, axis=1))
                squared_sum_item_desc = tf.reduce_sum(tf.square(tmp_item_desc), axis=1)
                fm_item_desc = 0.5 * (sum_squared_item_desc - squared_sum_item_desc)

                #### predict
                # concat
                deep_list += [
                    att_seq_name,
                    att_seq_item_desc,
                    context,
                    fm_name,
                    fm_item_desc,

                ]
                # if self.params["use_bigram"]:
                #     deep_list += [att_seq_bigram_item_desc]
                # # if self.params["use_trigram"]:
                # #     deep_list += [att_seq_trigram_item_desc]
                # if self.params["use_subword"]:
                #     deep_list += [att_seq_subword_item_desc]

            # fm layer
            fm_list = []
            if self.params["enable_fm_first_order"]:
                bias_seq_name = embed(self.seq_name, MAX_NUM_WORDS + 1, 1, reduce_sum=True,
                                      seed=self.params["random_seed"])
                bias_seq_item_desc = embed(self.seq_item_desc, MAX_NUM_WORDS + 1, 1, reduce_sum=True,
                                           seed=self.params["random_seed"])
                # bias_seq_category_name = embed(self.seq_category_name, MAX_NUM_WORDS + 1, 1, reduce_sum=True,
                #                                seed=self.params["random_seed"])
                if self.params["use_bigram"]:
                    bias_seq_bigram_item_desc = embed(self.seq_bigram_item_desc, MAX_NUM_BIGRAMS + 1, 1,
                                                      reduce_sum=True, seed=self.params["random_seed"])
                if self.params["use_trigram"]:
                    bias_seq_trigram_item_desc = embed(self.seq_trigram_item_desc, MAX_NUM_TRIGRAMS + 1, 1,
                                                       reduce_sum=True, seed=self.params["random_seed"])
                if self.params["use_subword"]:
                    bias_seq_subword_item_desc = embed(self.seq_subword_item_desc, MAX_NUM_SUBWORDS + 1, 1,
                                                       reduce_sum=True, seed=self.params["random_seed"])

                bias_brand_name = embed(self.brand_name, MAX_NUM_BRANDS, 1, flatten=True,
                                        seed=self.params["random_seed"])
                # bias_category_name = embed(self.category_name, MAX_NUM_CATEGORIES, 1, flatten=True)
                bias_category_name1 = embed(self.category_name1, MAX_NUM_CATEGORIES_LST[0], 1, flatten=True,
                                            seed=self.params["random_seed"])
                bias_category_name2 = embed(self.category_name2, MAX_NUM_CATEGORIES_LST[1], 1, flatten=True,
                                            seed=self.params["random_seed"])
                bias_category_name3 = embed(self.category_name3, MAX_NUM_CATEGORIES_LST[2], 1, flatten=True,
                                            seed=self.params["random_seed"])
                bias_item_condition = embed(self.item_condition_id, MAX_NUM_CONDITIONS + 1, 1, flatten=True,
                                            seed=self.params["random_seed"])
                bias_shipping = embed(self.shipping, MAX_NUM_SHIPPINGS, 1, flatten=True,
                                      seed=self.params["random_seed"])

                fm_first_order_list = [
                    bias_seq_name,
                    bias_seq_item_desc,
                    # bias_seq_category_name,
                    bias_brand_name,
                    # bias_category_name,
                    bias_category_name1,
                    bias_category_name2,
                    bias_category_name3,
                    bias_item_condition,
                    bias_shipping,
                ]
                if self.params["use_bigram"]:
                    fm_first_order_list += [bias_seq_bigram_item_desc]
                if self.params["use_trigram"]:
                    fm_first_order_list += [bias_seq_trigram_item_desc]
                # if self.params["use_subword"]:
                #     fm_first_order_list += [bias_seq_subword_item_desc]
                tmp_first_order = tf.concat(fm_first_order_list, axis=1)
                fm_list.append(tmp_first_order)

            if self.params["enable_fm_second_order"]:
                # second order
                emb_list = [
                    tf.expand_dims(att_seq_name, axis=1),
                    tf.expand_dims(att_seq_item_desc, axis=1),
                    # tf.expand_dims(att_seq_category_name, axis=1),

                    emb_brand_name,
                    # emb_category_name,
                    emb_category_name1,
                    emb_category_name2,
                    emb_category_name3,
                    emb_item_condition,
                    emb_shipping,

                ]
                if self.params["use_bigram"]:
                    emb_list += [tf.expand_dims(att_seq_bigram_item_desc, axis=1)]
                # if self.params["use_trigram"]:
                #     emb_list += [tf.expand_dims(att_seq_trigram_item_desc, axis=1)]
                if self.params["use_subword"]:
                    emb_list += [tf.expand_dims(att_seq_subword_item_desc, axis=1)]
                emb_concat = tf.concat(emb_list, axis=1)
                emb_sum_squared = tf.square(tf.reduce_sum(emb_concat, axis=1))
                emb_squared_sum = tf.reduce_sum(tf.square(emb_concat), axis=1)

                fm_second_order = 0.5 * (emb_sum_squared - emb_squared_sum)
                fm_list.extend([emb_sum_squared, emb_squared_sum])

            if self.params["enable_fm_second_order"] and self.params["enable_fm_higher_order"]:
                fm_higher_order = dense_block(fm_second_order, hidden_units=[self.params["embedding_dim"]] * 2,
                                              dropouts=[0.] * 2, densenet=False, training=self.training, seed=self.params["random_seed"])
                fm_list.append(fm_higher_order)

            if self.params["enable_deep"]:
                deep_list.extend(fm_list)
                deep_in = tf.concat(deep_list, axis=-1, name="concat")
                # dense
                hidden_units = [self.params["fc_dim"]*4, self.params["fc_dim"]*2, self.params["fc_dim"]]
                dropouts = [self.params["fc_dropout"]] * len(hidden_units)
                if self.params["fc_type"] == "fc":
                    deep_out = dense_block(deep_in, hidden_units=hidden_units, dropouts=dropouts, densenet=False,
                                           training=self.training, seed=self.params["random_seed"])
                elif self.params["fc_type"] == "resnet":
                    deep_out = resnet_block(deep_in, hidden_units=hidden_units, dropouts=dropouts, cardinality=1,
                                            dense_shortcut=True, training=self.training,
                                            seed=self.params["random_seed"])
                elif self.params["fc_type"] == "densenet":
                    deep_out = dense_block(deep_in, hidden_units=hidden_units, dropouts=dropouts, densenet=True,
                                           training=self.training, seed=self.params["random_seed"])
                fm_list.append(deep_out)


            fm_list.append(self.num_vars)
            fm_list.append(self.item_condition)
            fm_list.append(tf.cast(self.shipping, tf.float32))
            out = tf.concat(fm_list, axis=-1)


            self.pred = tf.layers.Dense(1, kernel_initializer=tf.glorot_uniform_initializer(self.params["random_seed"]),
                                        dtype=tf.float32, bias_initializer=tf.zeros_initializer())(out)

            # intermediate meta
            self.meta = out

            #### loss
            self.rmse = tf.sqrt(tf.losses.mean_squared_error(self.target, self.pred))
            # target is normalized, so std is 1
            # we apply 3 sigma principle
            std = 1.
            self.loss = tf.losses.huber_loss(self.target, self.pred, delta=1. * std)
            # self.rmse = tf.sqrt(self.loss)

            #### optimizer
            self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
            if self.params["optimizer_type"] == "nadam":
                self.optimizer = NadamOptimizer(learning_rate=self.learning_rate, beta1=self.params["beta1"],
                                                beta2=self.params["beta2"], epsilon=1e-8,
                                                schedule_decay=self.params["schedule_decay"])
            elif self.params["optimizer_type"] == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.params["beta1"],
                                                        beta2=self.params["beta2"], epsilon=1e-8)
            elif self.params["optimizer_type"] == "lazyadam":
                self.optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.learning_rate,
                                                                  beta1=self.params["beta1"],
                                                                  beta2=self.params["beta2"], epsilon=1e-8)
            elif self.params["optimizer_type"] == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-7)
            elif self.params["optimizer_type"] == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            elif self.params["optimizer_type"] == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95)
            elif self.params["optimizer_type"] == "rmsprop":
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.9, momentum=0.9,
                                                           epsilon=1e-8)
            elif self.params["optimizer_type"] == "powersign":
                self.optimizer = PowerSignOptimizer(learning_rate=self.learning_rate)
            elif self.params["optimizer_type"] == "addsign":
                self.optimizer = AddSignOptimizer(learning_rate=self.learning_rate)
            elif self.params["optimizer_type"] == "amsgrad":
                self.optimizer = AMSGradOptimizer(learning_rate=self.learning_rate, beta1=self.params["beta1"],
                                                  beta2=self.params["beta2"], epsilon=1e-8)

            #### training op
            """
            https://stackoverflow.com/questions/35803425/update-only-part-of-the-word-embedding-matrix-in-tensorflow
            TL;DR: The default implementation of opt.minimize(loss), TensorFlow will generate a sparse update for 
            word_emb that modifies only the rows of word_emb that participated in the forward pass.

            The gradient of the tf.gather(word_emb, indices) op with respect to word_emb is a tf.IndexedSlices object
             (see the implementation for more details). This object represents a sparse tensor that is zero everywhere, 
             except for the rows selected by indices. A call to opt.minimize(loss) calls 
             AdamOptimizer._apply_sparse(word_emb_grad, word_emb), which makes a call to tf.scatter_sub(word_emb, ...)* 
             that updates only the rows of word_emb that were selected by indices.

            If on the other hand you want to modify the tf.IndexedSlices that is returned by 
            opt.compute_gradients(loss, word_emb), you can perform arbitrary TensorFlow operations on its indices and 
            values properties, and create a new tf.IndexedSlices that can be passed to opt.apply_gradients([(word_emb, ...)]). 
            For example, you could cap the gradients using MyCapper() (as in the example) using the following calls:

            grad, = opt.compute_gradients(loss, word_emb)
            train_op = opt.apply_gradients(
                [tf.IndexedSlices(MyCapper(grad.values), grad.indices)])
            Similarly, you could change the set of indices that will be modified by creating a new tf.IndexedSlices with
             a different indices.

            * In general, if you want to update only part of a variable in TensorFlow, you can use the tf.scatter_update(), 
            tf.scatter_add(), or tf.scatter_sub() operators, which respectively set, add to (+=) or subtract from (-=) the 
            value previously stored in a variable.
            """
            # # it's slow
            # grads = self.optimizer.compute_gradients(self.loss)
            # for i, (g, v) in enumerate(grads):
            #     if g is not None:
            #         if isinstance(g, tf.IndexedSlices):
            #             grads[i] = (tf.IndexedSlices(tf.clip_by_norm(g.values, self.params["optimizer_clipnorm"]), g.indices), v)
            #         else:
            #             grads[i] = (tf.clip_by_norm(g, self.params["optimizer_clipnorm"]), v)
            # self.train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss)#, global_step=self.global_step)

            #### init
            self.sess, self.saver = self._init_session()

            # save model state to memory
            # https://stackoverflow.com/questions/46393983/how-can-i-restore-tensors-to-a-past-value-without-saving-the-value-to-disk/46511601
            # https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model/43333803#43333803
            # Extract the global varibles from the graph.
            self.gvars = self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            # Exract the Assign operations for later use.
            self.assign_ops = [self.graph.get_operation_by_name(v.op.name + "/Assign") for v in self.gvars]
            # Extract the initial value ops from each Assign op for later use.
            self.init_values = [assign_op.inputs[1] for assign_op in self.assign_ops]

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        # the following reduce the training time for a snapshot from 180~220s to 100s in kernel
        config.intra_op_parallelism_threads = 4
        config.inter_op_parallelism_threads = 4
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        # max_to_keep=None, keep all the models
        saver = tf.train.Saver(max_to_keep=None)
        return sess, saver

    def _save_session(self, dir):
        """Saves session = weights"""
        _makedirs(self.params["model_dir"])
        self.saver.save(self.sess, dir)

    def _restore_session(self, dir):
        self.saver.restore(self.sess, dir)

    def _save_state(self):
        # Record the current state of the TF global varaibles
        gvars_state = self.sess.run(self.gvars)
        self.gvars_state_list.append(gvars_state)

    def _restore_state(self, gvars_state):
        # Create a dictionary of the iniailizers and stored state of globals.
        feed_dict = dict(zip(self.init_values, gvars_state))
        # Use the initializer ops for each variable to load the stored values.
        self.sess.run(self.assign_ops, feed_dict=feed_dict)

    def _get_batch_index(self, seq, step):
        n = len(seq)
        res = []
        res_append = res.append
        for i in range(0, n, step):
            res_append(seq[i:i + step])
        # last batch
        if len(res) * step < n:
            res_append(seq[len(res) * step:])
        return res

    def _get_feed_dict(self, X, idx, dropout=0.1, training=False):
        feed_dict = {}
        # if self.params["use_bigram"]:
            # feed_dict[self.seq_bigram_name] = X["seq_bigram_name"][idx]
        feed_dict[self.seq_bigram_item_desc] = X["seq_bigram_item_desc"][idx]
            # if training and dropout > 0:
            #     mask = np.random.choice([0, 1], (len(idx), self.params["max_sequence_length_item_desc"]),
            #                             p=[dropout, 1 - dropout])
            #     feed_dict[self.seq_bigram_item_desc] *= mask
        # if self.params["use_trigram"]:
            # feed_dict[self.seq_trigram_name] = X["seq_trigram_name"][idx]
        feed_dict[self.seq_trigram_item_desc] = X["seq_trigram_item_desc"][idx]
        # if self.params["use_subword"]:
            # feed_dict[self.seq_subword_name] = X["seq_subword_name"][idx]
            # feed_dict[self.sequence_length_name_subword] = X["sequence_length_name_subword"][idx]
        # feed_dict[self.seq_subword_item_desc] = X["seq_subword_item_desc"][idx]
            # if training and dropout > 0:
            #     mask = np.random.choice([0, 1], (len(idx), self.params["max_sequence_length_item_desc_subword"]),
            #                             p=[dropout, 1 - dropout])
            #     feed_dict[self.seq_subword_item_desc] *= mask
        # if self.params["use_subword_list"]:
        #     feed_dict[self.seq_subword_list_item_desc] = X["seq_subword_list_item_desc"][idx]

        feed_dict.update({
            self.seq_name: X["seq_name"][idx],
            self.seq_item_desc: X["seq_item_desc"][idx],
            # self.seq_category_name: X["seq_category_name"][idx],
            self.brand_name: X["brand_name"][idx],
            # self.category_name: X["category_name"][idx],
            self.category_name1: X["category_name1"][idx],
            self.category_name2: X["category_name2"][idx],
            self.category_name3: X["category_name3"][idx],
            self.item_condition: X["item_condition"][idx],
            self.item_condition_id: X["item_condition_id"][idx],
            self.shipping: X["shipping"][idx],
            self.num_vars: X["num_vars"][idx],
            # len
        })

        # if training and dropout > 0:
        #     mask = np.random.choice([0, 1], (len(idx), self.params["max_sequence_length_name"]),
        #                             p=[dropout, 1 - dropout])
        #     feed_dict[self.seq_name] *= mask
        #     mask = np.random.choice([0, 1], (len(idx), self.params["max_sequence_length_item_desc"]),
        #                             p=[dropout, 1 - dropout])
        #     feed_dict[self.seq_item_desc] *= mask

        feed_dict.update({
            self.sequence_length_name: X["sequence_length_name"][idx],
            self.sequence_length_item_desc: X["sequence_length_item_desc"][idx],
            # self.sequence_length_category_name: X["sequence_length_category_name"][idx],
        })
        # if self.params["use_subword"]:
        feed_dict[self.sequence_length_item_desc_subword] = X["sequence_length_item_desc_subword"][idx]


        return feed_dict

    def fit(self, X, y, validation_data=None):
        y = y.reshape(-1, 1)
        start_time = time.time()
        l = y.shape[0]
        train_idx_shuffle = np.arange(l)
        epoch_best_ = 4
        rmsle_best_ = 10.
        cycle_num = 0
        decay_steps = self.params["first_decay_steps"]
        global_step = 0
        global_step_exp = 0
        global_step_total = 0
        snapshot_num = 0
        learning_rate_need_big_jump = False
        total_rmse = 0.
        rmse_decay = 0.9
        for epoch in range(self.params["epoch"]):
            print("epoch: %d" % (epoch + 1))
            np.random.seed(epoch)
            if snapshot_num >= self.params["snapshot_before_restarts"] and self.params["shuffle_with_replacement"]:
                train_idx_shuffle = np.random.choice(np.arange(l), l)
            else:
                np.random.shuffle(train_idx_shuffle)
            batches = self._get_batch_index(train_idx_shuffle, self.params["batch_size_train"])
            for i, idx in enumerate(batches):
                if snapshot_num >= self.params["max_snapshot_num"]:
                    break
                if learning_rate_need_big_jump:
                    learning_rate = self.params["lr_jump_rate"] * self.params["max_lr_exp"]
                    learning_rate_need_big_jump = False
                else:
                    learning_rate = self.params["max_lr_exp"]
                lr = _exponential_decay(learning_rate=learning_rate,
                                        global_step=global_step_exp,
                                        decay_steps=decay_steps,  # self.params["num_update_each_epoch"],
                                        decay_rate=self.params["lr_decay_each_epoch_exp"])
                feed_dict = self._get_feed_dict(X, idx, dropout=0.1, training=False)
                feed_dict[self.target] = y[idx]
                feed_dict[self.learning_rate] = lr
                feed_dict[self.training] = True
                rmse_, opt = self.sess.run((self.rmse, self.train_op), feed_dict=feed_dict)
                if RUNNING_MODE != "submission":
                    # scaling rmsle' = (1/scale_) * (raw rmsle)
                    # raw rmsle = scaling rmsle' * scale_
                    total_rmse = rmse_decay * total_rmse + (1. - rmse_decay) * rmse_ * (target_scaler.scale_)
                    logger.info("[batch-%d] train-rmsle=%.5f, lr=%.5f [%.1f s]" % (
                        i + 1, total_rmse,
                        lr, time.time() - start_time))
                # save model
                global_step += 1
                global_step_exp += 1
                global_step_total += 1
                if self.params["enable_snapshot_ensemble"]:
                    if global_step % decay_steps == 0:
                        cycle_num += 1
                        if cycle_num % self.params["snapshot_every_num_cycle"] == 0:
                            snapshot_num += 1
                            print("snapshot num: %d" % snapshot_num)
                            # self._save_session(dir=self.params["model_dir"] + "/%d/" % (snapshot_num))

                            # skip the first snapshot
                            # if snapshot_num > 2:# and snapshot_num != 5:
                            self._save_state()
                            logger.info("[model-%d] cycle num=%d, current lr=%.5f [%.5f]" % (
                                snapshot_num, cycle_num, lr, time.time() - start_time))
                            # reset global_step and first_decay_steps
                            decay_steps = self.params["first_decay_steps"]
                            if self.params["lr_jump_exp"] or snapshot_num >= self.params["snapshot_before_restarts"]:
                                learning_rate_need_big_jump = True
                        if snapshot_num >= self.params["snapshot_before_restarts"]:
                            global_step = 0
                            global_step_exp = 0
                            decay_steps *= self.params["t_mul"]

                if validation_data is not None and global_step_total % self.params["eval_every_num_update"] == 0:
                    y_pred = self._predict(validation_data[0])
                    y_valid_inv = target_scaler.inverse_transform(validation_data[1])
                    y_pred_inv = target_scaler.inverse_transform(y_pred)
                    rmsle = rmse(y_valid_inv, y_pred_inv)
                    logger.info("[step-%d] train-rmsle=%.5f, valid-rmsle=%.5f, lr=%.5f [%.1f s]" % (
                        global_step_total, total_rmse, rmsle, lr, time.time() - start_time))
                    if rmsle < rmsle_best_:
                        rmsle_best_ = rmsle
                        epoch_best_ = epoch + 1

        return rmsle_best_, epoch_best_

    def fit_predict(self, X, y, X_test, mode="mean"):
        y = y.reshape(-1, 1)
        start_time = time.time()
        l = y.shape[0]
        train_idx_shuffle = np.arange(l)
        cycle_num = 0
        decay_steps = self.params["first_decay_steps"]
        global_step = 0
        global_step_exp = 0
        global_step_total = 0
        snapshot_num = 0
        learning_rate_need_big_jump = False
        total_rmse = 0.
        rmse_decay = 0.9
        for epoch in range(self.params["epoch"]):
            print("epoch: %d" % (epoch + 1))
            np.random.seed(epoch)
            if snapshot_num >= self.params["snapshot_before_restarts"] and self.params["shuffle_with_replacement"]:
                train_idx_shuffle = np.random.choice(np.arange(l), l)
            else:
                np.random.shuffle(train_idx_shuffle)
            batches = self._get_batch_index(train_idx_shuffle, self.params["batch_size_train"])
            for i, idx in enumerate(batches):
                # if self.params["lr_schedule"] == "cosine_decay_restarts":
                if snapshot_num >= self.params["max_snapshot_num"]:
                    break
                if snapshot_num >= self.params["snapshot_before_restarts"]:
                    """
                    if learning_rate_need_big_jump:
                        learning_rate = self.params["lr_jump_rate"] * self.params["max_lr_cosine"]
                        learning_rate_need_big_jump = False
                    else:
                        learning_rate = self.params["max_lr_cosine"]
                    lr = _cosine_decay_restarts(learning_rate=learning_rate,
                                            global_step=global_step,
                                            first_decay_steps=decay_steps,
                                            t_mul=self.params["t_mul"],
                                            m_mul=self.params["m_mul"],
                                            alpha=self.params["base_lr"])
                    """
                    if learning_rate_need_big_jump:
                        learning_rate = self.params["lr_jump_rate"] * self.params["max_lr_exp"]
                        learning_rate_need_big_jump = False
                    else:
                        learning_rate = self.params["max_lr_exp"]
                    # elif self.params["lr_schedule"] == "exponential_decay":
                    lr = _exponential_decay(learning_rate=learning_rate,
                                            global_step=global_step_exp,
                                            decay_steps=decay_steps,  # self.params["num_update_each_epoch"],
                                            decay_rate=self.params["lr_decay_each_epoch_exp"])
                else:
                    if learning_rate_need_big_jump:
                        learning_rate = self.params["lr_jump_rate"] * self.params["max_lr_exp"]
                        learning_rate_need_big_jump = False
                    else:
                        learning_rate = self.params["max_lr_exp"]
                    # elif self.params["lr_schedule"] == "exponential_decay":
                    lr = _exponential_decay(learning_rate=learning_rate,
                                            global_step=global_step_exp,
                                            decay_steps=decay_steps,  # self.params["num_update_each_epoch"],
                                            decay_rate=self.params["lr_decay_each_epoch_exp"])
                feed_dict = self._get_feed_dict(X, idx, dropout=0.1, training=False)
                feed_dict[self.target] = y[idx]
                feed_dict[self.learning_rate] = lr
                feed_dict[self.training] = True
                rmse_, opt = self.sess.run((self.rmse, self.train_op), feed_dict=feed_dict)
                if RUNNING_MODE != "submission":
                    # scaling rmsle' = (1/scale_) * (raw rmsle)
                    # raw rmsle = scaling rmsle' * scale_
                    total_rmse = rmse_decay * total_rmse + (1. - rmse_decay) * rmse_ * (target_scaler.scale_)
                    logger.info("[batch-%d] train-rmsle=%.5f, lr=%.5f [%.1f s]" % (
                        i + 1, total_rmse,
                        lr, time.time() - start_time))
                global_step += 1
                global_step_exp += 1
                global_step_total += 1
                if self.params["enable_snapshot_ensemble"]:
                    # make prediction using snapshot ensemble
                    if global_step % decay_steps == 0:
                        cycle_num += 1
                        # global_step = 0
                        # global_step_exp = 0
                        # decay_steps *= self.params["t_mul"]
                        if cycle_num % self.params["snapshot_every_num_cycle"] == 0:
                            if snapshot_num == 0:
                                y_pred = self._predict(X_test)
                            else:
                                y_pred = np.hstack([y_pred, self._predict(X_test)])
                            snapshot_num += 1
                            print("snapshot num: %d" % snapshot_num)
                            if self.params["lr_jump_exp"] or snapshot_num >= self.params["snapshot_before_restarts"]:
                                learning_rate_need_big_jump = True
                        if snapshot_num >= self.params["snapshot_before_restarts"]:
                            global_step = 0
                            global_step_exp = 0
                            decay_steps *= self.params["t_mul"]
        if self.params["enable_snapshot_ensemble"]:
            if mode == "raw":
                pass
            elif mode == "mean":
                y_pred = np.mean(y_pred, axis=1, keepdims=True)
            elif mode == "median":
                y_pred = np.median(y_pred, axis=1, keepdims=True)
            elif mode == "weight":
                y_pred = self.bias + np.dot(y_pred, self.weights)
        else:
            y_pred = self._predict(X_test)
        return y_pred

    def _predict(self, X):
        l = X["seq_name"].shape[0]
        train_idx = np.arange(l)
        batches = self._get_batch_index(train_idx, self.params["batch_size_inference"])
        y = np.zeros((l, 1), dtype=np.float32)
        y_pred = []
        y_pred_append = y_pred.append
        for idx in batches:
            feed_dict = self._get_feed_dict(X, idx)
            feed_dict[self.target] = y[idx]
            feed_dict[self.learning_rate] = 1.0
            feed_dict[self.training] = False
            pred = self.sess.run((self.pred), feed_dict=feed_dict)
            y_pred_append(pred)
        y_pred = np.vstack(y_pred).reshape(-1, 1)
        return y_pred

    def _predict_meta_and_y(self, X):
        l = X["seq_name"].shape[0]
        train_idx = np.arange(l)
        batches = self._get_batch_index(train_idx, self.params["batch_size_inference"])
        y = np.zeros((l, 1), dtype=np.float32)
        meta = []
        y_pred = []
        y_pred_append = y_pred.append
        meta_append = meta.append
        for idx in batches:
            feed_dict = self._get_feed_dict(X, idx)
            feed_dict[self.target] = y[idx]
            feed_dict[self.learning_rate] = 1.0
            feed_dict[self.training] = False
            meta_, pred = self.sess.run((self.meta, self.pred), feed_dict=feed_dict)
            meta_append(meta_)
            y_pred_append(pred)
        meta = np.vstack(meta)
        y_pred = np.vstack(y_pred).reshape(-1, 1)
        return meta, y_pred

    def _merge_gvars_state_list(self):
        out = self.gvars_state_list[0].copy()
        for ms in self.gvars_state_list[1:]:
            for i, m in enumerate(ms):
                out[i] += m
        out = [o / float(len(self.gvars_state_list)) for o in out]
        return out

    def predict(self, X, mode="mean"):
        if self.params["enable_snapshot_ensemble"]:
            y = []
            if mode == "merge":
                gvars_state = self._merge_gvars_state_list()
                self._restore_state(gvars_state)
                y_ = self._predict(X)
                y.append(y_)
            else:
                for i,gvars_state in enumerate(self.gvars_state_list):
                    print("predict for: %d"%(i+1))
                    self._restore_state(gvars_state)
                    y_ = self._predict(X)
                    y.append(y_)
            if len(y) == 1:
                y = np.array(y).reshape(-1, 1)
            else:
                y = np.hstack(y)
                if mode == "median":
                    y = np.median(y, axis=1, keepdims=True)
                elif mode == "mean":
                    y = np.mean(y, axis=1, keepdims=True)
                elif mode == "weight":
                    y = self.bias + np.dot(y, self.weights)
                elif mode == "raw":
                    pass
        else:
            y = self._predict(X)

        return y


    def predict_meta_and_y(self, X, mode="mean"):
        if self.params["enable_snapshot_ensemble"]:
            meta = []
            y = []
            model_dirs = glob.glob("%s/[0-9]*/" % (self.params["model_dir"]))
            model_dirs = sorted(model_dirs, key=lambda x: int(x.split("/")[-2]))
            for model_dir in model_dirs:
                self._restore_session(model_dir)
                meta_, y_ = self._predict_meta_and_y(X)
                meta.append(meta_)
                y.append(y_)
            if len(y) == 1:
                meta = np.array(meta)
                y = np.array(y).reshape(-1, 1)
            else:
                meta = np.hstack(meta)
                y = np.hstack(y)
                if mode == "median":
                    meta = np.median(meta, axis=1, keepdims=True)
                    y = np.median(y, axis=1, keepdims=True)
                elif mode == "mean":
                    meta = np.mean(meta, axis=1, keepdims=True)
                    y = np.mean(y, axis=1, keepdims=True)
                elif mode == "raw":
                    pass
        else:
            meta, y = self._predict_meta_and_y(X)

        return meta, y


########################
# MODEL TRAINING
########################
"""
https://github.com/tensorflow/tensorflow/blob/3989529e6041be9b16009dd8b5b3889427b47952/tensorflow/python/training/learning_rate_decay.py
"""


def _exponential_decay(learning_rate, global_step, decay_steps, decay_rate,
                       staircase=False):
    p = global_step / decay_steps
    if staircase:
        p = np.floor(p)
    return learning_rate * np.power(decay_rate, p)


def cosine_decay(learning_rate, global_step, decay_steps, alpha=0.0,
                 name=None):
    """Applies cosine decay to the learning rate.
    See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent
    with Warm Restarts. https://arxiv.org/abs/1608.03983
    When training a model, it is often recommended to lower the learning rate as
    the training progresses.  This function applies a cosine decay function
    to a provided initial learning rate.  It requires a `global_step` value to
    compute the decayed learning rate.  You can just pass a TensorFlow variable
    that you increment at each training step.
    The function returns the decayed learning rate.  It is computed as:
    ```python
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    decayed_learning_rate = learning_rate * decayed
    ```
    Example usage:
    ```python
    decay_steps = 1000
    lr_decayed = cosine_decay(learning_rate, global_step, decay_steps)
    ```
    Args:
      learning_rate: A scalar `float32` or `float64` Tensor or a Python number.
        The initial learning rate.
      global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
        Global step to use for the decay computation.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Number of steps to decay over.
      alpha: A scalar `float32` or `float64` Tensor or a Python number.
        Minimum learning rate value as a fraction of learning_rate.
      name: String. Optional name of the operation.  Defaults to 'CosineDecay'.
    Returns:
      A scalar `Tensor` of the same type as `learning_rate`.  The decayed
      learning rate.
    Raises:
      ValueError: if `global_step` is not supplied.
    """
    if global_step is None:
        raise ValueError("cosine decay requires global_step")
    with ops.name_scope(name, "CosineDecay",
                        [learning_rate, global_step]) as name:
        learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        decay_steps = math_ops.cast(decay_steps, dtype)
        global_step = math_ops.minimum(global_step, decay_steps)
        completed_fraction = global_step / decay_steps
        cosine_decayed = 0.5 * (
                1.0 + math_ops.cos(constant_op.constant(math.pi) * completed_fraction))

        decayed = (1 - alpha) * cosine_decayed + alpha
        return math_ops.multiply(learning_rate, decayed)


def cosine_decay_restarts(learning_rate, global_step, first_decay_steps,
                          t_mul=2.0, m_mul=1.0, alpha=0.0, name=None):
    """Applies cosine decay with restarts to the learning rate.
    See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent
    with Warm Restarts. https://arxiv.org/abs/1608.03983
    When training a model, it is often recommended to lower the learning rate as
    the training progresses.  This function applies a cosine decay function with
    restarts to a provided initial learning rate.  It requires a `global_step`
    value to compute the decayed learning rate.  You can just pass a TensorFlow
    variable that you increment at each training step.
    The function returns the decayed learning rate while taking into account
    possible warm restarts. The learning rate multiplier first decays
    from 1 to `alpha` for `first_decay_steps` steps. Then, a warm
    restart is performed. Each new warm restart runs for `t_mul` times more steps
    and with `m_mul` times smaller initial learning rate.
    Example usage:
    ```python
    first_decay_steps = 1000
    lr_decayed = cosine_decay_restarts(learning_rate, global_step,
                                       first_decay_steps)
    ```
    Args:
      learning_rate: A scalar `float32` or `float64` Tensor or a Python number.
        The initial learning rate.
      global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
        Global step to use for the decay computation.
      first_decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Number of steps to decay over.
      t_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
        Used to derive the number of iterations in the i-th period
      m_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
        Used to derive the initial learning rate of the i-th period:
      alpha: A scalar `float32` or `float64` Tensor or a Python number.
        Minimum learning rate value as a fraction of the learning_rate.
      name: String. Optional name of the operation.  Defaults to 'SGDRDecay'.
    Returns:
      A scalar `Tensor` of the same type as `learning_rate`.  The decayed
      learning rate.
    Raises:
      ValueError: if `global_step` is not supplied.
    """
    if global_step is None:
        raise ValueError("cosine decay restarts requires global_step")
    with ops.name_scope(name, "SGDRDecay",
                        [learning_rate, global_step, first_decay_steps]) as name:
        learning_rate = ops.convert_to_tensor(learning_rate,
                                              name="initial_learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        first_decay_steps = math_ops.cast(first_decay_steps, dtype)
        alpha = math_ops.cast(alpha, dtype)
        t_mul = math_ops.cast(t_mul, dtype)
        m_mul = math_ops.cast(m_mul, dtype)

        completed_fraction = global_step / first_decay_steps

        def compute_step(completed_fraction, geometric=False):
            if geometric:
                i_restart = math_ops.floor(math_ops.log(1.0 - completed_fraction * (
                        1.0 - t_mul)) / math_ops.log(t_mul))

                sum_r = (1.0 - t_mul ** i_restart) / (1.0 - t_mul)
                completed_fraction = (completed_fraction - sum_r) / t_mul ** i_restart

            else:
                i_restart = math_ops.floor(completed_fraction)
                completed_fraction = completed_fraction - i_restart

            return i_restart, completed_fraction

        i_restart, completed_fraction = control_flow_ops.cond(
            math_ops.equal(t_mul, 1.0),
            lambda: compute_step(completed_fraction, geometric=False),
            lambda: compute_step(completed_fraction, geometric=True))

        m_fac = m_mul ** i_restart
        cosine_decayed = 0.5 * m_fac * (1.0 + math_ops.cos(
            constant_op.constant(math.pi) * completed_fraction))
        decayed = (1 - alpha) * cosine_decayed + alpha

    return math_ops.multiply(learning_rate, decayed, name=name)


def _cosine_decay_restarts(learning_rate, global_step, first_decay_steps,
                           t_mul=2.0, m_mul=1.0, alpha=0.0,
                           initial_variance=0.00, variance_decay=0.55):
    initial_variance = min(learning_rate, initial_variance / 2.)
    # noisy cosine decay with restarts
    completed_fraction = global_step / first_decay_steps

    def compute_step(completed_fraction, geometric=False):
        if geometric:
            i_restart = np.floor(np.log(1.0 - completed_fraction * (
                    1.0 - t_mul)) / np.log(t_mul))

            sum_r = (1.0 - t_mul ** i_restart) / (1.0 - t_mul)
            completed_fraction = (completed_fraction - sum_r) / t_mul ** i_restart

        else:
            i_restart = np.floor(completed_fraction)
            completed_fraction = completed_fraction - i_restart

        return i_restart, completed_fraction

    geometric = t_mul != 1.0
    i_restart, completed_fraction = compute_step(completed_fraction, geometric)
    m_fac = m_mul ** i_restart

    # # noise
    # variance = initial_variance / (np.power(1.0 + global_step, variance_decay))
    # std = np.sqrt(variance)
    # noisy_m_fac = m_fac + np.random.normal(0.0, std)
    noisy_m_fac = m_fac

    cosine_decayed = 0.5 * noisy_m_fac * (1.0 + np.cos(math.pi * completed_fraction))
    decayed = (1 - alpha) * cosine_decayed + alpha

    return learning_rate * decayed


def linear_cosine_decay(learning_rate, global_step, decay_steps,
                        num_periods=0.5, alpha=0.0, beta=0.001,
                        name=None):
    """Applies linear cosine decay to the learning rate.
    See [Bello et al., ICML2017] Neural Optimizer Search with RL.
    https://arxiv.org/abs/1709.07417
    For the idea of warm starts here controlled by `num_periods`,
    see [Loshchilov & Hutter, ICLR2016] SGDR: Stochastic Gradient Descent
    with Warm Restarts. https://arxiv.org/abs/1608.03983
    Note that linear cosine decay is more aggressive than cosine decay and
    larger initial learning rates can typically be used.
    When training a model, it is often recommended to lower the learning rate as
    the training progresses.  This function applies a linear cosine decay function
    to a provided initial learning rate.  It requires a `global_step` value to
    compute the decayed learning rate.  You can just pass a TensorFlow variable
    that you increment at each training step.
    The function returns the decayed learning rate.  It is computed as:
    ```python
    global_step = min(global_step, decay_steps)
    linear_decay = (decay_steps - global_step) / decay_steps)
    cosine_decay = 0.5 * (
        1 + cos(pi * 2 * num_periods * global_step / decay_steps))
    decayed = (alpha + linear_decay) * cosine_decay + beta
    decayed_learning_rate = learning_rate * decayed
    ```
    Example usage:
    ```python
    decay_steps = 1000
    lr_decayed = linear_cosine_decay(learning_rate, global_step, decay_steps)
    ```
    Args:
      learning_rate: A scalar `float32` or `float64` Tensor or a Python number.
        The initial learning rate.
      global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
        Global step to use for the decay computation.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Number of steps to decay over.
      num_periods: Number of periods in the cosine part of the decay.
        See computation above.
      alpha: See computation above.
      beta: See computation above.
      name: String.  Optional name of the operation.  Defaults to
        'LinearCosineDecay'.
    Returns:
      A scalar `Tensor` of the same type as `learning_rate`.  The decayed
      learning rate.
    Raises:
      ValueError: if `global_step` is not supplied.
    """
    if global_step is None:
        raise ValueError("linear cosine decay requires global_step")
    with ops.name_scope(name, "LinearCosineDecay",
                        [learning_rate, global_step]) as name:
        learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        decay_steps = math_ops.cast(decay_steps, dtype)
        num_periods = math_ops.cast(num_periods, dtype)
        global_step = math_ops.minimum(global_step, decay_steps)
        alpha = math_ops.cast(alpha, dtype)
        beta = math_ops.cast(beta, dtype)

        linear_decayed = (decay_steps - global_step) / decay_steps
        completed_fraction = global_step / decay_steps
        fraction = 2.0 * num_periods * completed_fraction
        cosine_decayed = 0.5 * (
                1.0 + math_ops.cos(constant_op.constant(math.pi) * fraction))

        linear_cosine_decayed = (alpha + linear_decayed) * cosine_decayed + beta
        return math_ops.multiply(learning_rate, linear_cosine_decayed, name=name)


def noisy_linear_cosine_decay(learning_rate, global_step, decay_steps,
                              initial_variance=1.0, variance_decay=0.55,
                              num_periods=0.5, alpha=0.0, beta=0.001,
                              name=None):
    """Applies noisy linear cosine decay to the learning rate.
    See [Bello et al., ICML2017] Neural Optimizer Search with RL.
    https://arxiv.org/abs/1709.07417
    For the idea of warm starts here controlled by `num_periods`,
    see [Loshchilov & Hutter, ICLR2016] SGDR: Stochastic Gradient Descent
    with Warm Restarts. https://arxiv.org/abs/1608.03983
    Note that linear cosine decay is more aggressive than cosine decay and
    larger initial learning rates can typically be used.
    When training a model, it is often recommended to lower the learning rate as
    the training progresses.  This function applies a noisy linear
    cosine decay function to a provided initial learning rate.
    It requires a `global_step` value to compute the decayed learning rate.
    You can just pass a TensorFlow variable that you increment at each
    training step.
    The function returns the decayed learning rate.  It is computed as:
    ```python
    global_step = min(global_step, decay_steps)
    linear_decay = (decay_steps - global_step) / decay_steps)
    cosine_decay = 0.5 * (
        1 + cos(pi * 2 * num_periods * global_step / decay_steps))
    decayed = (alpha + linear_decay + eps_t) * cosine_decay + beta
    decayed_learning_rate = learning_rate * decayed
    ```
    where eps_t is 0-centered gaussian noise with variance
    initial_variance / (1 + global_step) ** variance_decay
    Example usage:
    ```python
    decay_steps = 1000
    lr_decayed = noisy_linear_cosine_decay(
      learning_rate, global_step, decay_steps)
    ```
    Args:
      learning_rate: A scalar `float32` or `float64` Tensor or a Python number.
        The initial learning rate.
      global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
        Global step to use for the decay computation.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Number of steps to decay over.
      initial_variance: initial variance for the noise. See computation above.
      variance_decay: decay for the noise's variance. See computation above.
      num_periods: Number of periods in the cosine part of the decay.
        See computation above.
      alpha: See computation above.
      beta: See computation above.
      name: String.  Optional name of the operation.  Defaults to
        'NoisyLinearCosineDecay'.
    Returns:
      A scalar `Tensor` of the same type as `learning_rate`.  The decayed
      learning rate.
    Raises:
      ValueError: if `global_step` is not supplied.
    """
    if global_step is None:
        raise ValueError("noisy linear cosine decay requires global_step")
    with ops.name_scope(name, "NoisyLinearCosineDecay",
                        [learning_rate, global_step]) as name:
        learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        decay_steps = math_ops.cast(decay_steps, dtype)
        global_step = math_ops.minimum(global_step, decay_steps)
        initial_variance = math_ops.cast(initial_variance, dtype)
        variance_decay = math_ops.cast(variance_decay, dtype)
        num_periods = math_ops.cast(num_periods, dtype)
        alpha = math_ops.cast(alpha, dtype)
        beta = math_ops.cast(beta, dtype)

        linear_decayed = (decay_steps - global_step) / decay_steps
        variance = initial_variance / (
            math_ops.pow(1.0 + global_step, variance_decay))
        std = math_ops.sqrt(variance)
        noisy_linear_decayed = (
                linear_decayed + random_ops.random_normal(
            linear_decayed.shape, stddev=std))

        completed_fraction = global_step / decay_steps
        fraction = 2.0 * num_periods * completed_fraction
        cosine_decayed = 0.5 * (
                1.0 + math_ops.cos(constant_op.constant(math.pi) * fraction))
        noisy_linear_cosine_decayed = (
                (alpha + noisy_linear_decayed) * cosine_decayed + beta)

        return math_ops.multiply(
            learning_rate, noisy_linear_cosine_decayed, name=name)


def get_training_params(train_size, batch_size, params):
    params["num_update_each_epoch"] = int(train_size / float(batch_size))

    # # cyclic lr
    params["m_mul"] = np.power(params["lr_decay_each_epoch_cosine"], 1. / params["num_cycle_each_epoch"])
    params["m_mul_exp"] = np.power(params["lr_decay_each_epoch_exp"], 1. / params["num_cycle_each_epoch"])
    if params["t_mul"] == 1:
        tmp = int(params["num_update_each_epoch"] / params["num_cycle_each_epoch"])
    else:
        tmp = int(params["num_update_each_epoch"] / params["snapshot_every_epoch"] * (1. - params["t_mul"]) / (
                1. - np.power(params["t_mul"], params["num_cycle_each_epoch"] / params["snapshot_every_epoch"])))
    params["first_decay_steps"] = max([tmp, 1])
    params["snapshot_every_num_cycle"] = params["num_cycle_each_epoch"] // params["snapshot_every_epoch"]
    params["snapshot_every_num_cycle"] = max(params["snapshot_every_num_cycle"], 1)

    # cnn
    if params["cnn_timedistributed"]:
        params["cnn_num_filters"] = params["embedding_dim"]

    # text dim after the encode step
    if params["encode_method"] == "fasttext":
        encode_text_dim = params["embedding_dim"]
    elif params["encode_method"] == "textcnn":
        encode_text_dim = params["cnn_num_filters"] * len(params["cnn_filter_sizes"])
    elif params["encode_method"] in ["textrnn", "textbirnn"]:
        encode_text_dim = params["rnn_num_units"]
    elif params["encode_method"] == "fasttext+textcnn":
        encode_text_dim = params["embedding_dim"] + params["cnn_num_filters"] * len(
            params["cnn_filter_sizes"])
    elif params["encode_method"] in ["fasttext+textrnn", "fasttext+textbirnn"]:
        encode_text_dim = params["embedding_dim"] + params["rnn_num_units"]
    elif params["encode_method"] in ["fasttext+textcnn+textrnn", "fasttext+textcnn+textbirnn"]:
        encode_text_dim = params["embedding_dim"] + params["cnn_num_filters"] * len(
            params["cnn_filter_sizes"]) + params["rnn_num_units"]
    params["encode_text_dim"] = encode_text_dim

    return params


def cross_validation_hyperopt(dfTrain, params):
    global target_scaler
    params = ModelParamSpace()._convert_int_param(params)
    _print_param_dict(params)
    rmsle_best_ = None
    epoch_best_ = None

    # level1, valid index
    level1Ratio, validRatio = 0.6, 0.4
    num_train = dfTrain.shape[0]
    level1Size = int(level1Ratio * num_train)
    indices = np.arange(num_train)
    np.random.seed(params["random_seed"])
    np.random.shuffle(indices)
    level1Ind, validInd = indices[:level1Size], indices[level1Size:]
    y_level1, y_valid = dfTrain.price.values[level1Ind].reshape((-1, 1)), dfTrain.price.values[validInd].reshape(
        (-1, 1))
    y_valid_inv = target_scaler.inverse_transform(y_valid)

    # keras =======================
    X_level1, lbs, params = get_tf_data(dfTrain.iloc[level1Ind], lbs=None, params=params)
    X_valid, lbs, _ = get_tf_data(dfTrain.iloc[validInd], lbs=lbs, params=params)

    params = get_training_params(train_size=len(level1Ind), batch_size=params["batch_size_train"],
                                 params=params)
    model = MercariNet(params)
    model.fit(X_level1, y_level1, validation_data=(X_valid, y_valid))
    y_valid_tf = model.predict(X_valid, mode="raw")
    y_valid_tf_inv = target_scaler.inverse_transform(y_valid_tf)
    for j in reversed(range(y_valid_tf.shape[1])):
        rmsle = rmse(y_valid_inv, y_valid_tf_inv[:, j, np.newaxis])
        logger.info("valid-rmsle (tf of last %d): %.5f" % (y_valid_tf.shape[1] - j, rmsle))
        y_valid_tf_inv_ = np.mean(y_valid_tf_inv[:, j:], axis=1, keepdims=True)
        rmsle = rmse(y_valid_inv, y_valid_tf_inv_)
        logger.info(
            "valid-rmsle (tf snapshot ensemble with mean of last %d): %.5f" % (y_valid_tf.shape[1] - j, rmsle))

    stacking_model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=4)
    stacking_model.fit(y_valid_tf, y_valid)
    logger.info(stacking_model.intercept_)
    logger.info(stacking_model.coef_)
    y_valid_stack = stacking_model.predict(y_valid_tf).reshape((-1, 1))
    y_valid_stack_inv = target_scaler.inverse_transform(y_valid_stack)
    rmsle = rmse(y_valid_inv, y_valid_stack_inv)
    logger.info("rmsle (stack): %.5f" % rmsle)
    rmsle_best_ = rmsle

    # """
    rmsle_mean = rmsle if rmsle_best_ is None else rmsle_best_
    rmsle_std = 0
    epoch_best = params["epoch"] if epoch_best_ is None else epoch_best_
    logger.info("RMSLE")
    logger.info("      Mean: %.6f" % rmsle_mean)
    logger.info("      Std: %.6f" % rmsle_std)
    ret = {
        "loss": rmsle_mean,
        "attachments": {
            "std": rmsle_std,
            "epoch": epoch_best,
        },
        "status": STATUS_OK,
    }
    return ret


def submission(params):
    global MAX_NUM_BRANDS
    global MAX_NUM_CATEGORIES
    global MAX_NUM_CATEGORIES_LST
    global MAX_NUM_CONDITIONS
    global MAX_NUM_SHIPPINGS
    global target_scaler

    params = ModelParamSpace()._convert_int_param(params)
    _print_param_dict(params)
    start_time = time.time()

    dfTrain = load_train_data()
    target_scaler = MyStandardScaler()
    dfTrain["price"] = target_scaler.fit_transform(dfTrain["price"].values.reshape(-1, 1))
    dfTrain, word_index, bigram_index, trigram_index, subword_index, subword_list_index, label_encoder = preprocess(
        dfTrain)

    X_train, lbs_tf, params = get_tf_data(dfTrain, lbs=None, params=params)
    y_train = dfTrain.price.values.reshape((-1, 1))

    MAX_NUM_BRANDS = dfTrain["brand_name_cat"].max() + 1
    MAX_NUM_CATEGORIES = dfTrain["category_name_cat"].max() + 1
    MAX_NUM_CATEGORIES_LST = [0] * MAX_CATEGORY_NAME_LEN
    for i in range(MAX_CATEGORY_NAME_LEN):
        MAX_NUM_CATEGORIES_LST[i] = dfTrain["category_name%d_cat" % (i + 1)].max() + 1
    MAX_NUM_CONDITIONS = dfTrain["item_condition_id"].max()
    MAX_NUM_SHIPPINGS = 2

    del dfTrain
    gc.collect()
    print('[%.5f] Finished loading data' % (time.time() - start_time))

    params = get_training_params(train_size=len(y_train), batch_size=params["batch_size_train"], params=params)
    model = MercariNet(params)
    model.fit(X_train, y_train)
    del X_train
    del y_train
    gc.collect()
    print('[%.5f] Finished training tf' % (time.time() - start_time))

    y_test = []
    id_test = []
    for dfTest in load_test_data(chunksize=350000*2):
        dfTest, _, _, _, _, _, _ = preprocess(dfTest, word_index, bigram_index, trigram_index, subword_index,
                                                 subword_list_index, label_encoder)
        X_test, lbs_tf, _ = get_tf_data(dfTest, lbs=lbs_tf, params=params)

        y_test_ = model.predict(X_test, mode="weight")
        y_test.append(y_test_)
        id_test.append(dfTest.id.values.reshape((-1, 1)))

    y_test = np.vstack(y_test)
    id_test = np.vstack(id_test)
    y_test = np.expm1(target_scaler.inverse_transform(y_test))
    y_test = y_test.flatten()
    id_test = id_test.flatten()
    id_test = id_test.astype(int)
    y_test[y_test < 0.0] = 0.0
    submission = pd.DataFrame({"test_id": id_test, "price": y_test})
    submission.to_csv("sample_submission.csv", index=False)
    print('[%.5f] Finished prediction' % (time.time() - start_time))


# -------------------------------------- fasttext ---------------------------------------------
def _print_param_dict(d, prefix="      ", incr_prefix="      "):
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            logger.info("%s%s:" % (prefix, k))
            _print_param_dict(v, prefix + incr_prefix, incr_prefix)
        else:
            logger.info("%s%s: %s" % (prefix, k, v))


class ModelParamSpace:
    def __init__(self):
        pass

    def _convert_int_param(self, param_dict):
        if isinstance(param_dict, dict):
            for k, v in param_dict.items():
                if k in int_params:
                    param_dict[k] = v if v is None else int(v)
                elif isinstance(v, list) or isinstance(v, tuple):
                    for i in range(len(v)):
                        self._convert_int_param(v[i])
                elif isinstance(v, dict):
                    self._convert_int_param(v)
        return param_dict


if RUNNING_MODE == "validation":
    load_data_success = False
    pkl_file = "../input/dfTrain_bigram_[MAX_NUM_WORDS_%d]_[MAX_NUM_BIGRAMS_%d]_[VOCAB_HASHING_TRICK_%s].pkl" % (
        MAX_NUM_WORDS, MAX_NUM_BIGRAMS, str(VOCAB_HASHING_TRICK))
    if USE_PREPROCESSED_DATA:
        try:
            with open(pkl_file, "rb") as f:
                dfTrain = pkl.load(f)
            if DEBUG:
                dfTrain = dfTrain.head(DEBUG_SAMPLE_NUM)
            load_data_success = True
        except:
            pass
    if not load_data_success:
        dfTrain = load_train_data()
        dfTrain, word_index, bigram_index, trigram_index, subword_index, subword_list_index, label_encoder = preprocess(
            dfTrain)
    target_scaler = MyStandardScaler()
    dfTrain["price"] = target_scaler.fit_transform(dfTrain["price"].values.reshape(-1, 1))

    MAX_NUM_BRANDS = dfTrain["brand_name_cat"].max() + 1
    MAX_NUM_CATEGORIES = dfTrain["category_name_cat"].max() + 1
    MAX_NUM_CATEGORIES_LST = [0] * MAX_CATEGORY_NAME_LEN
    for i in range(MAX_CATEGORY_NAME_LEN):
        MAX_NUM_CATEGORIES_LST[i] = dfTrain["category_name%d_cat" % (i + 1)].max() + 1
    MAX_NUM_CONDITIONS = dfTrain["item_condition_id"].max()
    MAX_NUM_SHIPPINGS = 2

    start_time = time.time()
    trials = Trials()
    obj = lambda param: cross_validation_hyperopt(dfTrain, param)
    best = fmin(obj, param_space_hyperopt, tpe.suggest, HYPEROPT_MAX_EVALS, trials)
    best_params = space_eval(param_space_hyperopt, best)
    best_params = ModelParamSpace()._convert_int_param(best_params)
    trial_rmsles = np.asarray(trials.losses(), dtype=float)
    best_ind = np.argmin(trial_rmsles)
    best_rmse_mean = trial_rmsles[best_ind]
    best_rmse_std = trials.trial_attachments(trials.trials[best_ind])["std"]
    best_epoch = trials.trial_attachments(trials.trials[best_ind])["epoch"]
    logger.info("-" * 50)
    logger.info("Best RMSLE")
    logger.info("      Mean: %.6f" % best_rmse_mean)
    logger.info("      Std: %.6f" % best_rmse_std)
    logger.info("      Epoch: %d" % best_epoch)
    logger.info("Best param")
    _print_param_dict(best_params)
    logger.info("time: %.5f" % (time.time() - start_time))

else:
    submission(param_space_best)