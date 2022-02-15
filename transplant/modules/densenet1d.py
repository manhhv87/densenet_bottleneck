import tensorflow as tf


class _DenseBlock(tf.keras.layers.Layer):
    """
        Creates a convolution block consisting of BN-ReLU-Conv.
        Optional: bottleneck, dropout
    """
    def __init__(self, growth_rate, kernel_size, dropout_rate=None, bottleneck=False, **kwargs):  # constructor
        super().__init__(**kwargs)
        self.growth_rate = growth_rate
        self.kernel_size = kernel_size
        self.bottleneck = bottleneck
        self.dropout_rate = dropout_rate
        self.listLayers = []

    def build(self, input_shape):
        # Bottleneck
        if self.bottleneck:
            self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
            self.relu = tf.keras.layers.Activation('relu')
            self.conv = tf.keras.layers.Conv1D(4 * self.growth_rate, kernel_size=1, use_bias=False,
                                               kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-5))
            self.listLayers.append(self.bn)
            self.listLayers.append(self.relu)
            self.listLayers.append(self.conv)

            # Dropout
            if self.dropout_rate:
                self.drop = tf.keras.layers.Dropout(self.dropout_rate)
                self.listLayers.append(self.drop)

        # Standard (BN-ReLU-Conv)
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu1 = tf.keras.layers.Activation('relu')
        self.conv1 = tf.keras.layers.Conv1D(filters=self.growth_rate, kernel_size=self.kernel_size,
                                            padding='same', use_bias=False,
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-5))
        self.listLayers.append(self.bn1)
        self.listLayers.append(self.relu1)
        self.listLayers.append(self.conv1)

        # Dropout
        if self.dropout_rate:
            self.drop1 = tf.keras.layers.Dropout(self.dropout_rate)
            self.listLayers.append(self.drop1)

        super().build(input_shape)

    def call(self, x, **kwargs):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x, y], axis=-1)
        return y


class _TransitionBlock(tf.keras.layers.Layer):
    """
        Creates a transition layer between dense blocks as transition, which do convolution and pooling.
        Works as down-sampling.
    """
    def __init__(self, nb_channels, dropout_rate=None, **kwargs):
        super().__init__(**kwargs)
        self.nb_channels = nb_channels
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu = tf.keras.layers.Activation('relu')
        self.conv = tf.keras.layers.Conv1D(self.nb_channels, padding='same', kernel_size=1, use_bias=False,
                                           kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                           kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-5))

        # Adding dropout
        if self.dropout_rate:
            self.drop = tf.keras.layers.Dropout(self.dropout_rate)

        self.avg_pool = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)

        super().build(input_shape)

    def call(self, x, **kwargs):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)

        # Adding dropout
        if self.dropout_rate:
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

    def __init__(self, input_shape=(16384, 1), num_outputs=1, blocks=(6, 12, 24, 16), first_num_channels=64, growth_rate=32,
                 kernel_size=(3, 3, 3, 3), block_fn1=_DenseBlock, block_fn2=_TransitionBlock,
                 bottleneck=False, dropout_rate=None, include_top=True, **kwargs):  # constructor

        super().__init__(**kwargs)
        self.input_layer = tf.keras.layers.Input(shape=input_shape)

        # Built Convolution layer
        self.conv = tf.keras.layers.Conv1D(filters=first_num_channels, kernel_size=7, padding='same',
                                           strides=2, use_bias=False,
                                           kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                           kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-5))
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu = tf.keras.layers.Activation('relu')
        self.maxpool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')

        # Built Dense Blocks and Transition layers
        self.densenet_blocks = []
        num_channel_trans = first_num_channels
        for stage, _ in enumerate(blocks):  # stage = [0,1,2,3] and _ = [6, 12, 24, 16]
            for block in range(blocks[stage]):
                self.densenet_blocks.append(block_fn1(growth_rate=growth_rate,
                                                      kernel_size=kernel_size[stage],
                                                      dropout_rate=dropout_rate,
                                                      bottleneck=bottleneck))

            # This is the number of output channels in the previous dense block
            num_channel_trans += blocks[stage] * growth_rate

            # A transition layer that halves the number of channels is added
            # between the dense blocks
            if stage != len(blocks) - 1:
                num_channel_trans //= 2
                self.densenet_blocks.append(block_fn2(nb_channels=num_channel_trans, dropout_rate=dropout_rate))

        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu1 = tf.keras.layers.Activation('relu')

        # include top layer (full connected layer)
        self.include_top = include_top
        if include_top:
            # average pool, 1-d fc, sigmoid
            self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
            out_act = 'sigmoid' if num_outputs == 1 else 'softmax'
            self.classifier = tf.keras.layers.Dense(num_outputs, out_act)

        self.out = self.call(self.input_layer)

    def call(self, x, include_top=None, **kwargs):
        if include_top is None:
            include_top = self.include_top

        # Built conv1 layer
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Built other layers
        for dnet_block in self.densenet_blocks:
            x = dnet_block(x)

        x = self.bn1(x)
        x = self.relu1(x)

        # include top layer (full connected layer)
        if include_top:
            x = self.global_pool(x)
            x = self.classifier(x)
        return x

    def model(self):
        # x = tf.keras.layers.Input(shape=(16384, 1))
        return tf.keras.Model(inputs=[self.in_shape], outputs=self.call(self.in_shape))
