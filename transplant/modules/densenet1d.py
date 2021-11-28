import tensorflow as tf


def batch_norm():
    return tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)


def relu():
    # return tf.keras.layers.ReLU()
    # return tf.keras.layers.ELU()
    return tf.keras.layers.LeakyReLU()
    # return tf.keras.layers.PReLU()


def conv1d(filters, kernel_size=1, strides=1):
    return tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,
                                  padding='same', use_bias=False,
                                  kernel_initializer=tf.keras.initializers.he_uniform())
                                  # kernel_initializer = tf.keras.initializers.he_normal())
                                  # kernel_initializer=tf.keras.initializers.VarianceScaling())

class _DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, bottleneck=True, dropout_rate=None, **kwargs):  # constructor
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.bottleneck = bottleneck
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        if self.bottleneck:
            self.bn = batch_norm()
            self.relu = relu()
            # self.conv = conv1d(filters=4 * self.num_filters)
            self.conv = conv1d(filters=self.num_filters)

            if self.dropout_rate is not None:
                self.drop = tf.keras.layers.Dropout(rate=self.dropout_rate)

        self.bn1 = batch_norm()
        self.relu1 = relu()
        self.conv1 = conv1d(filters=self.num_filters, kernel_size=self.kernel_size)

        if self.dropout_rate is not None:
            self.drop1 = tf.keras.layers.Dropout(rate=self.dropout_rate)

        if self.bottleneck and self.dropout_rate:
            self.listLayers = [self.bn, self.relu, self.conv, self.drop, self.bn1, self.relu1, self.conv1, self.drop1]
        elif self.bottleneck and not self.dropout_rate:
            self.listLayers = [self.bn, self.relu, self.conv, self.bn1, self.relu1, self.conv1]
        elif not self.bottleneck and self.dropout_rate:
            self.listLayers = [self.bn1, self.relu1, self.conv1, self.drop1]
        else:
            self.listLayers = [self.bn1, self.relu1, self.conv1]

        super().build(input_shape)

    def call(self, x, **kwargs):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x, y], axis=-1)
        return y


class _TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters, dropout_rate=None, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.bn = batch_norm()
        self.relu = relu()
        self.conv = conv1d(self.num_filters)

        if self.dropout_rate is not None:
            self.drop = tf.keras.layers.Dropout(rate=self.dropout_rate)

        self.avg_pool = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='same')
        super().build(input_shape)

    def call(self, x, **kwargs):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)

        if self.dropout_rate is not None:
            x = self.drop(x)

        return self.avg_pool(x)


class _DenseNet(tf.keras.Model):
    """"
    Densenet-BC model class, based "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>

    Args:
        num_outputs (int) - number of classification classes
        blocks (list of 3 or 4 ints) - how many dense layers in each dense block
        first_num_channels (int) - the number of filters to learn in the first convolution layer
        growth_rate (int) - how many filters to add each dense-layer (`k` in paper)
        block_fn1 - dense block
        block_fn2 - transition block
        include_top (bool) - yes or no include top layer
    """

    def __init__(self, num_outputs=1, blocks=(6, 12, 24, 16), first_num_channels=64, growth_rate=32,
                 kernel_size=(3, 3, 3, 3), dropout_rate=None, block_fn1=_DenseBlock, block_fn2=_TransitionBlock,
                 bottleneck=True, include_top=True, **kwargs):  # constructor

        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate

        # Built Convolution layer
        self.conv = conv1d(filters=first_num_channels, kernel_size=7, strides=2)  # 7×7, 64, stride 2
        self.bn = batch_norm()
        self.relu = relu()

        if self.dropout_rate is not None:
            self.drop = tf.keras.layers.Dropout(rate=self.dropout_rate)

        self.maxpool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')  # 3×3 max pool, stride 2

        # Built Dense Blocks and Transition layers
        self.densenet_blocks = []
        num_channel_trans = first_num_channels
        for stage, _ in enumerate(blocks):  # stage = [0,1,2,3] and _ = [6, 12, 24, 16]
            for block in range(blocks[stage]):
                dnet_block = block_fn1(num_filters=growth_rate, kernel_size=kernel_size[stage], bottleneck=bottleneck)
                self.densenet_blocks.append(dnet_block)

            # This is the number of output channels in the previous dense block
            num_channel_trans += blocks[stage] * growth_rate

            # A transition layer that halves the number of channels is added
            # between the dense blocks
            if stage != len(blocks) - 1:
                num_channel_trans //= 2
                tran_block = block_fn2(num_filters=num_channel_trans)
                self.densenet_blocks.append(tran_block)

        # include top layer (full connected layer)
        self.include_top = include_top
        if include_top:
            # average pool, 1-d fc, sigmoid
            self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
            out_act = 'sigmoid' if num_outputs == 1 else 'softmax'
            self.classifier = tf.keras.layers.Dense(num_outputs, out_act)

    def call(self, x, include_top=None, **kwargs):
        if include_top is None:
            include_top = self.include_top

        # Built conv1 layer
        x = self.conv(x)

        if self.dropout_rate is not None:
            x = self.drop(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Built other layers
        for dnet_block in self.densenet_blocks:
            x = dnet_block(x)

        # include top layer (full connected layer)
        if include_top:
            x = self.global_pool(x)
            x = self.classifier(x)
        return x
