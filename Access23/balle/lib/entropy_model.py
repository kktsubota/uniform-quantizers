import argparse
import scipy.stats

import tensorflow.compat.v1 as tf
from tensorflow.python.keras.engine import input_spec
from tensorflow_compression.python.ops import range_coding_ops
import tensorflow_compression as tfc

from lib.context_3d import get_network_cls
from lib.ops import quantize, _quantize1d, transpose_NCHW_to_NHWC, transpose_NHWC_to_NCHW
from lib.quantizer import DSQ, NonUQ


class RoundingEntropyBottleneck(tfc.EntropyBottleneck):
    def __init__(
        self,
        init_scale=10,
        filters=(3, 3, 3),
        data_format="channels_last",
        approx="STE-Q",
        sub_mean: bool = True,
        stop_gradient: bool = False,
        **kwargs
    ):
        super(RoundingEntropyBottleneck, self).__init__(
            init_scale=init_scale, filters=filters, data_format=data_format, **kwargs
        )
        self.approx = approx
        self.sub_mean = sub_mean
        self.tau = 0.5
        self.tau2 = 0.5
        self.T = 1.0
        self.k = 1.0
        self.sigma = 1.0
        self.n_centers = 6
        if self.approx == "DS-Q":
            self.quantizer = DSQ(self.k_init)
        elif self.approx == "NonU-Q":
            self.quantizer = NonUQ(self.n_centers, self.sigma)

    def _quantize(self, inputs, mode):
        _, _, _, input_slices = self._get_input_dims()
        medians = self._medians[input_slices]

        if mode == "noise":
            if not self.sub_mean:
                medians = None
            if self.approx in {"DS-Q", "NonU-Q"}:
                assert medians is None
                return self.quantizer(inputs)
            else:
                return quantize(inputs, medians, method=self.approx, tau=self.tau, tau2=self.tau2, T=self.T, k=self.k)

        outputs = tf.math.round(inputs - medians)
        outputs = tf.cast(outputs, self.dtype)
        if mode == "dequantize":
            return outputs + medians
        else:
            assert mode == "symbols", mode
            outputs = tf.cast(outputs, tf.int32)
            return outputs

    def _likelihood(self, inputs):
        if self.stop_gradient:
            inputs = tf.stop_gradient(inputs)
        if self.approx == "NonU-Q":
            ndim, channel_axis, _, _ = self._get_input_dims()

            # Convert to (channels, 1, batch) format by commuting channels to front
            # and then collapsing.
            order = list(range(ndim))
            order.pop(channel_axis)
            order.insert(0, channel_axis)
            inputs = tf.transpose(inputs, order)
            shape = tf.shape(inputs)
            inputs = tf.reshape(inputs, (shape[0], 1, -1))

            # Evaluate densities of centers (c_0, c_1, c_2, c_3, c_4, c_5)
            inputs_bound = (self.quantizer.centers[1:] + self.quantizer.centers[:-1]) / 2
            # (c0+c1)/2, ..., (c4+c5)/2
            bound = self._logits_cumulative(inputs_bound, stop_gradient=False)
            lower = bound[:-1]
            upper = bound[1:]

            # Flip signs if we can move more towards the left tail of the sigmoid.
            sign = -tf.math.sign(tf.math.add_n([lower, upper]))
            sign = tf.stop_gradient(sign)

            likelihood_bound = abs(
                tf.math.sigmoid(sign * upper) - tf.math.sigmoid(sign * lower))
            # add [-inf, (c0+c1)/2] and [(c4+c5)/2, inf])]
            likelihood_c = tf.concat([tf.sigmoid(bound[0]), likelihood_bound, 1 - tf.sigmoid(bound[-1])], axis=0)
            
            # similarities to the centers
            similarity_c = tf.nn.softmax(tf.nn.l2(inputs - self.quantizer.centers) * self.quantizer.sigma)
            likelihood_soft = similarity_c @ likelihood_c
            likelihood = likelihood_soft

            # likelihood_hard = tf.one_hot(tf.argmax(similarity_c, axis=-1)) @ likelihood_c
            # likelihood = tf.stop_gradient(likelihood_hard - likelihood_soft) + likelihood_soft

            # Convert back to input tensor shape.
            order = list(range(1, ndim))
            order.insert(channel_axis, 0)
            likelihood = tf.reshape(likelihood, shape)
            likelihood = tf.transpose(likelihood, order)

            return likelihood
        else:
            return super(RoundingEntropyBottleneck, self)._likelihood(inputs)


class RoundingGaussianConditional(tfc.SymmetricConditional):
    # SRA-Q
    tau: float = 0.5
    # SGA-Q
    tau2: float = 0.5
    # sigmoid
    T: float = 1.0
    # DSfix-Q, DS-Q
    k: float = 1.0

    def __init__(self, scale, scale_table,
                scale_bound=None, mean=None, indexes=None, approx: str = "AUN-Q", sub_mean: bool = True, **kwargs):
        super().__init__(scale, scale_table, scale_bound=scale_bound, mean=mean, indexes=indexes, **kwargs)
        self.approx = approx
        self.sub_mean = sub_mean

    def _quantize(self, inputs, mode):
        if mode == "noise":
            if not self.sub_mean or self.mean is None:
                medians = None
            else:
                medians = self.mean
            return quantize(inputs, medians, method=self.approx, tau=self.tau, tau2=self.tau2, T=self.T, k=self.k)
        
        if self.mean is not None:
            inputs -= self.mean
        outputs = tf.math.round(inputs)
        outputs = tf.cast(outputs, self.dtype)
        if mode == "dequantize":
            if self.mean is not None:
                outputs += self.mean
            return outputs
        else:
            assert mode == "symbols", mode
            outputs = tf.cast(outputs, tf.int32)
            return outputs

    # copy from GaussianConditional
    def _standardized_cumulative(self, inputs):
        half = tf.constant(.5, dtype=self.dtype)
        const = tf.constant(-(2 ** -0.5), dtype=self.dtype)
        # Using the complementary error function maximizes numerical precision.
        return half * tf.math.erfc(const * inputs)

    def _standardized_quantile(self, quantile):
        return scipy.stats.norm.ppf(quantile)


class Config:
    # INVALID config (for 'base')
    # arch = 'base'
    # kernel_size = 5
    # INVALID config
    # lr_initial = 1e-4  # initial learning rate
    # optimizer = "ADAM"
    # lr_schedule = "DECAY"
    # lr_schedule_decay_interval = 2  # num epochs before decay
    # lr_schedule_decay_rate = 0.1
    # lr_schedule_decay_staircase = True

    arch_param__k = 24
    arch_param__non_linearity = 'relu'
    # NON USED config
    # arch_param__fc = 64

    regularization_factor = None
    learn_pad_var = False

    # rewrite config for 'res-shallow'
    kernel_size = 3
    arch = 'res_shallow'

    # add settings for centers by copying from ae_config.
    centers_initial_range = (-2, 2)
    num_centers = 6
    regularization_factor_centers = 0.1

    # manually added
    # stop_gradient of entropy models
    stop_gradient = False
    sigma = 1.0
    
    # NOTE: equivalent to MergeConfig in omegaconf to some extent
    def __init__(self, args: argparse.Namespace = None):
        if args is None:
            return
        
        if hasattr(args, "arch_param__k") and args.arch_param__k is not None:
            self.arch_param__k = args.arch_param__k
        
        if hasattr(args, "num_centers") and args.num_centers is not None:
            self.num_centers = args.num_centers
        
        if hasattr(args, "regularization_factor_centers") and args.regularization_factor_centers is not None:
            self.regularization_factor_centers = args.regularization_factor_centers

        if hasattr(args, "stop_gradient") and args.stop_gradient is not None:
            self.stop_gradient = args.stop_gradient

        if hasattr(args, "sigma") and args.sigma is not None:
            self.sigma = args.sigma


class NonUQEntropyModel(tf.keras.layers.Layer):
    def __init__(self, pc_config: Config, **kwargs):
        super(NonUQEntropyModel, self).__init__(**kwargs)
        pc_cls = get_network_cls(pc_config)

        n_centers: int = pc_config.num_centers
        self.pc = pc_cls(pc_config, num_centers=n_centers)
        self.stop_gradient = pc_config.stop_gradient
        self.sigma = pc_config.sigma

        vmin, vmax = pc_config.centers_initial_range
        initializer = tf.random_uniform_initializer(minval=vmin, maxval=vmax, seed=666)
        self.centers = self.add_weight(name="centers", shape=(n_centers,), initializer=initializer, trainable=True)

    def call(self, inputs, training: bool):
        assert self.centers is not None
        qbar = self._quantize(inputs, mode="noise", sigma=self.sigma)
        # the gradient is not backpropagated through the quantization
        target_symbols = self._quantize(inputs, mode="symbols", sigma=self.sigma)

        # The gradient to the encoder is stopped in the code of [Mentzer+, CVPR 18].
        # https://github.com/fab-jul/imgcomp-cvpr/blob/master/code/train.py#L104
        if self.stop_gradient:
            pc_in = tf.stop_gradient(qbar)
        else:
            pc_in = qbar

        # likelihood?
        # self.pc.auto_pad_value(ae)
        pc_in = transpose_NHWC_to_NCHW(pc_in)
        target_symbols = transpose_NHWC_to_NCHW(target_symbols)
        bc_train = self.pc.bitcost(
            pc_in, target_symbols, is_training=True, pad_value=self.centers[0]
        )
        bc_train = transpose_NCHW_to_NHWC(bc_train)
        return qbar, bc_train
    
    def _quantize(self, inputs, mode: str, sigma: float = 1.0):
        # return qsot, qhard, symbols
        qsoft, qhard, symbols = _quantize1d(inputs, self.centers, sigma, data_format='NHWC')

        assert mode in {"noise", "dequantize", "symbols"}
        if mode == "noise":
            with tf.name_scope('qbar'):
                qbar = qsoft + tf.stop_gradient(qhard - qsoft)
            return qbar
        elif mode == "dequantize":
            return qhard
        elif mode == "symbols":
            return symbols
    
    def _dequantize(self, inputs, mode: str):
        assert mode == "dequantize"
        phi_hard = tf.one_hot(inputs, depth=len(self.centers), axis=-1, dtype=tf.float32)
        return phi_hard
    
    def _likelihood(self, inputs):
        raise NotImplementedError
        # before softmax activation
        # return self.pc.logits(inputs, is_training=True)

    def compress(self, inputs):
        """Compress inputs and store their binary representations into strings.

        Arguments:
        inputs: `Tensor` with values to be compressed.

        Returns:
        compressed: String `Tensor` vector containing the compressed
            representation of each batch element of `inputs`.

        Raises:
        ValueError: if `inputs` has an integral or inconsistent `DType`, or
            inconsistent number of channels.
        """
        raise NotImplementedError
        with tf.name_scope(self._name_scope()):
            inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
            if not self.built:
                # Check input assumptions set before layer building, e.g. input rank.
                input_spec.assert_input_compatibility(
                    self.input_spec, inputs, self.name)
                if self.dtype is None:
                    self._dtype = inputs.dtype.base_dtype.name
                self.build(inputs.shape)

            # Check input assumptions set after layer building, e.g. input shape.
            if not tf.executing_eagerly():
                input_spec.assert_input_compatibility(
                    self.input_spec, inputs, self.name)
                if inputs.dtype.is_integer:
                    raise ValueError(
                        "{} can't take integer inputs.".format(type(self).__name__))

            symbols = self._quantize(inputs, "symbols")
            assert symbols.dtype == tf.int32

            ndim = self.input_spec.ndim
            indexes = self._prepare_indexes(shape=tf.shape(symbols)[1:])
            broadcast_indexes = (indexes.shape.ndims != ndim)
            if broadcast_indexes:
                # We can't currently broadcast over anything else but the batch axis.
                assert indexes.shape.ndims == ndim - 1
                args = (symbols,)
            else:
                args = (symbols, indexes)

            # def loop_body(args):
            #     string = range_coding_ops.unbounded_index_range_encode(
            #         args[0], indexes if broadcast_indexes else args[1],
            #         self._quantized_cdf, self._cdf_length, self._offset,
            #         precision=self.range_coder_precision, overflow_width=4,
            #         debug_level=0)
            #     return string

            # strings = tf.map_fn(
            #     loop_body, args, dtype=tf.string,
            #     back_prop=False, name="compress")

            strings = tf.empty(symbols.shape)
            for i in range(symbols.shape[1]):
                for j in range(symbol.shape[2]):
                    for k in range(symbols.shape[3]):
                        symbol = range_coding_ops.unbounded_index_range_decode(
                            strings[:, i, j, k], indexes if broadcast_indexes else args[1],
                            self._quantized_cdf, self._cdf_length, self._offset,
                            precision=self.range_coder_precision, overflow_width=4,
                            debug_level=0)
                        strings[:, i:i+1, j:j+1, k:k+1] = self._dequantize(symbol[:, None, None, None], "dequantize")


            if not tf.executing_eagerly():
                strings.set_shape(inputs.shape[:1])

            return strings

    def decompress(self, strings, **kwargs):
        """Decompress values from their compressed string representations.

        Arguments:
        strings: A string `Tensor` vector containing the compressed data.
        **kwargs: Model-specific keyword arguments.

        Returns:
        The decompressed `Tensor`.
        """
        raise NotImplementedError
        with tf.name_scope(self._name_scope()):
            strings = tf.convert_to_tensor(strings, dtype=tf.string)

            indexes = self._prepare_indexes(**kwargs)
            ndim = self.input_spec.ndim
            broadcast_indexes = (indexes.shape.ndims != ndim)
            if broadcast_indexes:
                # We can't currently broadcast over anything else but the batch axis.
                assert indexes.shape.ndims == ndim - 1
                args = (strings,)
            else:
                args = (strings, indexes)

            # original
            # def loop_body(args):
            #     symbols = range_coding_ops.unbounded_index_range_decode(
            #         args[0], indexes if broadcast_indexes else args[1],
            #         self._quantized_cdf, self._cdf_length, self._offset,
            #         precision=self.range_coder_precision, overflow_width=4,
            #         debug_level=0)
            #     return symbols

            # symbols = tf.map_fn(
            #     loop_body, args, dtype=tf.int32, back_prop=False, name="decompress")
            # outputs = self._dequantize(symbols, "dequantize")
            
            # ours
            outputs = tf.empty(strings.shape)
            for i in range(symbols.shape[1]):
                for j in range(symbol.shape[2]):
                    for k in range(symbols.shape[3]):
                        symbol = range_coding_ops.unbounded_index_range_decode(
                            strings[:, i, j, k], indexes if broadcast_indexes else args[1],
                            self._quantized_cdf, self._cdf_length, self._offset,
                            precision=self.range_coder_precision, overflow_width=4,
                            debug_level=0)
                        outputs[:, i:i+1, j:j+1, k:k+1] = self._dequantize(symbol[:, None, None, None], "dequantize")
                        # update: self._quantized_cdf, self._cdf_length, self._offset using outputs

            assert outputs.dtype == self.dtype

            if not tf.executing_eagerly():
                outputs.set_shape(self.input_spec.shape)

            return outputs
