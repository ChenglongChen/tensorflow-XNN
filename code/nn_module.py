
import tensorflow as tf

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

