import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim as slim
import tensorflow_compression as tfc

from lib.layer import NonLocalAttentionBlock, residual_block_mentzer


class ThreeConvEncoder(tf.keras.layers.Layer):
    """The analysis transform."""

    def __init__(self, num_filters, latent_size=None, *args, **kwargs):
        self.num_filters = num_filters
        self.latent_size = num_filters if latent_size is None else latent_size
        super(ThreeConvEncoder, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters,
                (9, 9),
                name="layer_0",
                corr=True,
                strides_down=4,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_0"),
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_1",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_1"),
            ),
            tfc.SignalConv2D(
                self.latent_size,
                (5, 5),
                name="layer_2",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=False,
                activation=None,
            ),
        ]
        super(ThreeConvEncoder, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class FourConvEncoder(tf.keras.layers.Layer):
    """The analysis transform."""

    def __init__(self, num_filters, latent_size=None, *args, **kwargs):
        self.num_filters = num_filters
        self.latent_size = num_filters if latent_size is None else latent_size
        super(FourConvEncoder, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_0",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_0"),
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_1",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_1"),
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_2",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_2"),
            ),
            tfc.SignalConv2D(
                self.latent_size,
                (5, 5),
                name="layer_3",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=None,
            ),
        ]
        super(FourConvEncoder, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class Cheng20Encoder:
    def __init__(self, num_filters: int, latent_size=None, shallow: bool = False):
        self.num_filters = num_filters
        self.latent_size = num_filters if latent_size is None else latent_size
        self.shallow = shallow
        
    def __call__(self, tensor):
        if self.shallow:
            with tf.variable_scope("analysis"):
                for i in range(4):
                    with tf.variable_scope(f"layer_{i}"):
                        layer = tfc.SignalConv2D(
                            self.num_filters,
                            (5, 5),
                            corr=True,
                            strides_down=2,
                            padding="same_zeros",
                            use_bias=True,
                            activation=tfc.GDN(name=f"gdn_{i}") if i < 3 else None,
                        )
                        tensor = layer(tensor)
                return tensor

        kernel_size = 3
        # Use three 3x3 filters to replace one 9x9

        with tf.variable_scope("analysis"):

            # Four down-sampling blocks
            for i in range(4):
                if i > 0:
                    with tf.variable_scope("Block_" + str(i) + "_layer_0"):
                        layer = tfc.SignalConv2D(
                            self.num_filters,
                            (kernel_size, kernel_size),
                            corr=True,
                            strides_down=1,
                            padding="same_zeros",
                            use_bias=True,
                            activation=tf.nn.leaky_relu,
                            name="signal_conv2d",
                        )
                        tensor2 = layer(tensor)

                    with tf.variable_scope("Block_" + str(i) + "_layer_1"):
                        layer = tfc.SignalConv2D(
                            self.num_filters,
                            (kernel_size, kernel_size),
                            corr=True,
                            strides_down=1,
                            padding="same_zeros",
                            use_bias=True,
                            activation=tf.nn.leaky_relu,
                            name="signal_conv2d",
                        )
                        tensor2 = layer(tensor2)

                    tensor = tensor + tensor2

                if i < 3:
                    with tf.variable_scope("Block_" + str(i) + "_shortcut"):
                        shortcut = tfc.SignalConv2D(
                            self.num_filters,
                            (1, 1),
                            corr=True,
                            strides_down=2,
                            padding="same_zeros",
                            use_bias=True,
                            activation=None,
                            name="signal_conv2d",
                        )
                        shortcut_tensor = shortcut(tensor)

                    with tf.variable_scope("Block_" + str(i) + "_layer_2"):
                        layer = tfc.SignalConv2D(
                            self.num_filters,
                            (kernel_size, kernel_size),
                            corr=True,
                            strides_down=2,
                            padding="same_zeros",
                            use_bias=True,
                            activation=tf.nn.leaky_relu,
                            name="signal_conv2d",
                        )
                        tensor = layer(tensor)

                    with tf.variable_scope("Block_" + str(i) + "_layer_3"):
                        layer = tfc.SignalConv2D(
                            self.num_filters,
                            (kernel_size, kernel_size),
                            corr=True,
                            strides_down=1,
                            padding="same_zeros",
                            use_bias=True,
                            activation=tfc.GDN(name="gdn"),
                            name="signal_conv2d",
                        )
                        tensor = layer(tensor)

                        tensor = tensor + shortcut_tensor

                    if i == 1:
                        # Add one NLAM
                        tensor = NonLocalAttentionBlock(tensor, self.num_filters, scope="NLAB_0")

                else:
                    with tf.variable_scope("Block_" + str(i) + "_layer_2"):
                        layer = tfc.SignalConv2D(
                            self.latent_size,
                            (kernel_size, kernel_size),
                            corr=True,
                            strides_down=2,
                            padding="same_zeros",
                            use_bias=False,
                            activation=None,
                            name="signal_conv2d",
                        )
                        tensor = layer(tensor)

                    # Add one NLAM
                    tensor = NonLocalAttentionBlock(tensor, self.latent_size, scope="NLAB_1")

            return tensor


class Mentzer18Encoder:
    # https://github.com/fab-jul/imgcomp-cvpr/blob/master/code/ae_configs/cvpr/base#L31
    arch_param_B: int = 5

    def __init__(self, num_filters, latent_size=None):
        self.num_filters = num_filters
        self.latent_size = num_filters if latent_size is None else latent_size
    
    # remove heatmap and input normalization from
    # https://github.com/fab-jul/imgcomp-cvpr/blob/master/code/autoencoder.py#L218-L244
    def __call__(self, x):
        n = self.num_filters
        net = x

        net = slim.conv2d(net, n // 2, [5, 5], stride=2, scope='h1')
        net = slim.conv2d(net, n, [5, 5], stride=2, scope='h2')
        residual_input_0 = net
        for b in range(self.arch_param_B):
            residual_input_b = net
            with tf.variable_scope('res_block_enc_{}'.format(b)):
                net = residual_block_mentzer(net, n, num_conv2d=2, kernel_size=[3, 3], scope='enc_{}_1'.format(b))
                net = residual_block_mentzer(net, n, num_conv2d=2, kernel_size=[3, 3], scope='enc_{}_2'.format(b))
                net = residual_block_mentzer(net, n, num_conv2d=2, kernel_size=[3, 3], scope='enc_{}_3'.format(b))
            net = net + residual_input_b
        net = residual_block_mentzer(net, n, num_conv2d=2, kernel_size=[3, 3], scope='res_block_enc_final',
                                activation_fn=None)
        net = net + residual_input_0
        net = slim.conv2d(net, n, [5, 5], stride=2, activation_fn=None, scope='to_bn')
        return net
