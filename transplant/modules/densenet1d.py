import tensorflow as tf


def batch_norm():
    return tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)


def relu():
    return tf.keras.layers.ReLU()


def conv1d(filters, kernel_size=3, strides=1):
    return tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,
                                  padding='same', use_bias=False,
                                  kernel_initializer=tf.keras.initializers.VarianceScaling())  # initial weights matrix


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):  # constructor
        super().__init__(**kwargs)
        self.num_channels = num_channels

    def build(self, input_shape):
        self.bn = batch_norm()
        self.relu = relu()
        self.conv = conv1d(filters=self.num_channels, kernel_size=1)
        self.bn1 = batch_norm()
        self.relu1 = relu()
        self.conv1 = conv1d(filters=self.num_channels)

        self.listLayers = [self.bn, self.relu, self.conv, self.bn1, self.relu1, self.conv1]
        super().build(input_shape)

    def call(self, x, **kwargs):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x, y], axis=-1)
        return y


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels, **kwargs):  # constructor
        super().__init__(**kwargs)
        self.num_convs = num_convs
        self.num_channels = num_channels

    def build(self, input_shape):
        self.listLayers = []
        for _ in range(self.num_convs):
            self.listLayers.append(ConvBlock(self.num_channels))
        super().build(input_shape)

    def call(self, x, **kwargs):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x


class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels

    def build(self, input_shape):
        self.bn = batch_norm()
        self.relu = relu()
        self.conv = conv1d(self.num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool1D(pool_size=2, strides=2)
        super().build(input_shape)

    def call(self, x, **kwargs):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)


class DenseNet(tf.keras.Model):
    def __init__(self, num_outputs=1, num_convs_in_dense_blocks=(4, 4, 4, 4),
                 first_num_channels=64, growth_rate=(32, 32, 32, 32),
                 block_fn1=DenseBlock, block_fn2=TransitionBlock,
                 include_top=True, **kwargs):  # constructor

        super().__init__(**kwargs)

        # Built Convolution layer
        self.conv1 = conv1d(filters=64, kernel_size=7, strides=2)  # 7×7, 64, stride 2
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.maxpool1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same')  # 3×3 max pool, stride 2

        # Built Dense Blocks and Transition layers
        self.blocks = []
        num_channels = first_num_channels
        for stage, _ in enumerate(num_convs_in_dense_blocks):  # stage = [0,1,2,3] and _=[4,4,4,4]
            dnet_block = block_fn1(num_convs_in_dense_blocks[stage], growth_rate[stage])
            self.blocks.append(dnet_block)

            # This is the number of output channels in the previous dense block
            num_channels += num_convs_in_dense_blocks[stage] * growth_rate[stage]

            # A transition layer that halves the number of channels is added
            # between the dense blocks
            if stage != len(num_convs_in_dense_blocks) - 1:
                num_channels //= 2
                tran_block = block_fn2(num_channels)
                self.blocks.append(tran_block)

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
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.maxpool1(x)

        # Built other layers
        for dnet_block in self.blocks:
            x = dnet_block(x)

        # include top layer (full connected layer)
        if include_top:
            x = self.global_pool(x)
            x = self.classifier(x)
        return x
