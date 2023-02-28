import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim as slim
import tensorflow_compression as tfc

from lib.layer import residual_block_mentzer, NonLocalAttentionBlock


class ThreeConvDecoder(tf.keras.layers.Layer):
    """The synthesis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(ThreeConvDecoder, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_0",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_0", inverse=True),
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_1",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_1", inverse=True),
            ),
            tfc.SignalConv2D(
                3,
                (9, 9),
                name="layer_2",
                corr=False,
                strides_up=4,
                padding="same_zeros",
                use_bias=True,
                activation=None,
            ),
        ]
        super(ThreeConvDecoder, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class FourConvDecoder(tf.keras.layers.Layer):
    """The synthesis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(FourConvDecoder, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_0",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_0", inverse=True),
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_1",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_1", inverse=True),
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_2",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_2", inverse=True),
            ),
            tfc.SignalConv2D(
                3,
                (5, 5),
                name="layer_3",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=None,
            ),
        ]
        super(FourConvDecoder, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor



class Cheng20Decoder:
    def __init__(self, num_filters, shallow: bool = False):
        self.num_filters = num_filters
        self.shallow = shallow

    def __call__(self, tensor):
        if self.shallow:
            with tf.variable_scope("synthesis"):
                for i in range(4):
                    with tf.variable_scope(f"layer_{i}"):
                        layer = tfc.SignalConv2D(
                            self.num_filters if i < 3 else 3,
                            (5, 5),
                            corr=False,
                            strides_up=2,
                            padding="same_zeros",
                            use_bias=True,
                            activation=tfc.GDN(name=f"igdn_{i}", inverse=True) if i < 3 else None,
                        )
                        tensor = layer(tensor)
                return tensor


        kernel_size = 3
        # Use four 3x3 filters to replace one 9x9

        with tf.variable_scope("synthesis"):

            # Four up-sampling blocks
            for i in range(4):
                if i == 0:
                    # Add one NLAM
                    tensor = NonLocalAttentionBlock(tensor, self.num_filters, scope="NLAB_0")

                if i == 2:
                    # Add one NLAM
                    tensor = NonLocalAttentionBlock(tensor, self.num_filters, scope="NLAB_1")

                with tf.variable_scope("Block_" + str(i) + "_layer_0"):
                    layer = tfc.SignalConv2D(
                        self.num_filters,
                        (kernel_size, kernel_size),
                        corr=False,
                        strides_up=1,
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
                        corr=False,
                        strides_up=1,
                        padding="same_zeros",
                        use_bias=True,
                        activation=tf.nn.leaky_relu,
                        name="signal_conv2d",
                    )
                    tensor2 = layer(tensor2)
                    tensor = tensor + tensor2

                if i < 3:
                    with tf.variable_scope("Block_" + str(i) + "_shortcut"):

                        # Use Sub-Pixel to replace deconv.
                        shortcut = tfc.SignalConv2D(
                            self.num_filters * 4,
                            (1, 1),
                            corr=False,
                            strides_up=1,
                            padding="same_zeros",
                            use_bias=True,
                            activation=None,
                            name="signal_conv2d",
                        )
                        shortcut_tensor = shortcut(tensor)
                        shortcut_tensor = tf.depth_to_space(shortcut_tensor, 2)

                    with tf.variable_scope("Block_" + str(i) + "_layer_2"):

                        # Use Sub-Pixel to replace deconv.
                        layer = tfc.SignalConv2D(
                            self.num_filters * 4,
                            (kernel_size, kernel_size),
                            corr=False,
                            strides_up=1,
                            padding="same_zeros",
                            use_bias=True,
                            activation=tf.nn.leaky_relu,
                            name="signal_conv2d",
                        )
                        tensor = layer(tensor)
                        tensor = tf.depth_to_space(tensor, 2)

                    with tf.variable_scope("Block_" + str(i) + "_layer_3"):
                        layer = tfc.SignalConv2D(
                            self.num_filters,
                            (kernel_size, kernel_size),
                            corr=False,
                            strides_up=1,
                            padding="same_zeros",
                            use_bias=True,
                            activation=tfc.GDN(name="igdn", inverse=True),
                            name="signal_conv2d",
                        )
                        tensor = layer(tensor)

                        tensor = tensor + shortcut_tensor

                else:
                    with tf.variable_scope("Block_" + str(i) + "_layer_2"):

                        # Use Sub-Pixel to replace deconv.
                        layer = tfc.SignalConv2D(
                            12,
                            (kernel_size, kernel_size),
                            corr=False,
                            strides_up=1,
                            padding="same_zeros",
                            use_bias=True,
                            activation=None,
                            name="signal_conv2d",
                        )
                        tensor = layer(tensor)
                        tensor = tf.depth_to_space(tensor, 2)

            return tensor


class Mentzer18Decoder:
    # https://github.com/fab-jul/imgcomp-cvpr/blob/master/code/ae_configs/cvpr/base#L31
    arch_param_B: int = 5

    def __init__(self, num_filters):
        self.num_filters = num_filters

    # https://github.com/fab-jul/imgcomp-cvpr/blob/master/code/autoencoder.py#L246-L268
    def __call__(self, q):
        n = self.num_filters
        fa = 3
        fb = 5

        net = slim.conv2d_transpose(q, n, [fa, fa], stride=2, scope='from_bn')
        residual_input_0 = net
        for b in range(self.arch_param_B):
            residual_input_b = net
            with tf.variable_scope('res_block_dec_{}'.format(b)):
                net = residual_block_mentzer(net, n, num_conv2d=2, kernel_size=[3, 3], scope='dec_{}_1'.format(b))
                net = residual_block_mentzer(net, n, num_conv2d=2, kernel_size=[3, 3], scope='dec_{}_2'.format(b))
                net = residual_block_mentzer(net, n, num_conv2d=2, kernel_size=[3, 3], scope='dec_{}_3'.format(b))
            net = net + residual_input_b
        net = residual_block_mentzer(net, n, num_conv2d=2, kernel_size=[3, 3], scope='dec_after_res',
                            activation_fn=None)
        net = net + residual_input_0

        net = slim.conv2d_transpose(net, n // 2, [fb, fb], stride=2, scope='h12')
        net = slim.conv2d_transpose(net, 3, [fb, fb], stride=2, scope='h13', activation_fn=None)
        # net = tf.clip_by_value(net, 0, 1.0, name='clip')
        return net
