import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim as slim
import tensorflow_compression as tfc

# https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention/blob/master/network.py
def residualblock(tensor, num_filters, scope="residual_block"):
    """Builds the residual block"""
    with tf.variable_scope(scope):
        with tf.variable_scope("conv0"):
            layer = tfc.SignalConv2D(
                num_filters // 2,
                (1, 1),
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.relu,
                name="signal_conv2d",
            )
            output = layer(tensor)

        with tf.variable_scope("conv1"):
            layer = tfc.SignalConv2D(
                num_filters // 2,
                (3, 3),
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.relu,
                name="signal_conv2d",
            )
            output = layer(output)

        with tf.variable_scope("conv2"):
            layer = tfc.SignalConv2D(
                num_filters,
                (1, 1),
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=None,
                name="signal_conv2d",
            )
            output = layer(output)

        tensor = tensor + output

    return tensor

# https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention/blob/master/network.py
def NonLocalAttentionBlock(input_x, num_filters, scope="NonLocalAttentionBlock"):
    """Builds the non-local attention block"""
    with tf.variable_scope(scope):
        trunk_branch = residualblock(input_x, num_filters, scope="trunk_RB_0")
        trunk_branch = residualblock(trunk_branch, num_filters, scope="trunk_RB_1")
        trunk_branch = residualblock(trunk_branch, num_filters, scope="trunk_RB_2")

        attention_branch = residualblock(input_x, num_filters, scope="attention_RB_0")
        attention_branch = residualblock(
            attention_branch, num_filters, scope="attention_RB_1"
        )
        attention_branch = residualblock(
            attention_branch, num_filters, scope="attention_RB_2"
        )

        with tf.variable_scope("conv_1x1"):
            layer = tfc.SignalConv2D(
                num_filters,
                (1, 1),
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=None,
                name="signal_conv2d",
            )
            attention_branch = layer(attention_branch)
        attention_branch = tf.sigmoid(attention_branch)

    tensor = input_x + tf.multiply(attention_branch, trunk_branch)
    return tensor


@slim.add_arg_scope
def residual_block_mentzer(x, num_outputs, num_conv2d, **kwargs):
    assert 'num_outputs' not in kwargs
    kwargs['num_outputs'] = num_outputs

    residual_input = x
    with tf.variable_scope(kwargs.get('scope', None), 'res'):
        for conv_i in range(num_conv2d):
            kwargs['scope'] = 'conv{}'.format(conv_i + 1)
            if conv_i == (num_conv2d - 1):  # no relu after final conv
                kwargs['activation_fn'] = None
            x = slim.conv2d(x, **kwargs)

        return x + residual_input


# class ResidualBlock(tf.keras.layers.Layer):
#     """The analysis transform."""

#     def __init__(self, num_filters, *args, **kwargs):
#         self.num_filters = num_filters
#         super(ResidualBlock, self).__init__(*args, **kwargs)

#     def build(self, input_shape):
#         self._layers = [
#             tfc.SignalConv2D(
#                 self.num_filters // 2,
#                 (1, 1),
#                 corr=True,
#                 strides_down=1,
#                 padding="same_zeros",
#                 use_bias=True,
#                 activation=tf.nn.relu,
#                 name="signal_conv2d",
#             ),
#             tfc.SignalConv2D(
#                 self.num_filters // 2,
#                 (3, 3),
#                 corr=True,
#                 strides_down=1,
#                 padding="same_zeros",
#                 use_bias=True,
#                 activation=tf.nn.relu,
#                 name="signal_conv2d",
#             ),
#             tfc.SignalConv2D(
#                 self.num_filters,
#                 (1, 1),
#                 corr=True,
#                 strides_down=1,
#                 padding="same_zeros",
#                 use_bias=True,
#                 activation=None,
#                 name="signal_conv2d",
#             ),
#         ]
#         super(ResidualBlock, self).build(input_shape)

#     def call(self, tensor):
#         output = tensor
#         for layer in self._layers:
#             output = layer(output)
#         return tensor + output


# class NonLocalAttentionBlock(tf.keras.layers.Layer):
#     """Builds the non-local attention block"""

#     def __init__(self, num_filters, *args, **kwargs):
#         self.num_filters = num_filters
#         super(NonLocalAttentionBlock, self).__init__(*args, **kwargs)

#     def build(self, input_shape):
#         self.trunk_layers = [
#             ResidualBlock(self.num_filters, name="trunk_RB_0"),
#             ResidualBlock(self.num_filters, name="trunk_RB_1"),
#             ResidualBlock(self.num_filters, name="trunk_RB_2"),
#         ]

#         self.attention_layers = [
#             ResidualBlock(self.num_filters, name="attention_RB_0"),
#             ResidualBlock(self.num_filters, name="attention_RB_1"),
#             ResidualBlock(self.num_filters, name="attention_RB_2"),
#             tfc.SignalConv2D(
#                 self.num_filters,
#                 (1, 1),
#                 corr=True,
#                 strides_down=1,
#                 padding="same_zeros",
#                 use_bias=True,
#                 activation=tf.nn.sigmoid,
#                 name="signal_conv2d",
#             ),
#         ]
#         super(NonLocalAttentionBlock, self).build(input_shape)

#     def call(self, tensor):
#         attn = tensor
#         for layer in self.attention_layers:
#             attn = layer(attn)

#         trunk = tensor
#         for layer in self.trunk_layers:
#             trunk = layer(trunk)

#         return tensor + tf.multiply(attn, trunk)
