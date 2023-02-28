#!/usr/bin/env python3
import os
import numpy as np

import tensorflow as tf
import tensorflow.compat.v1.distributions as tfd
import tensorflow_compression as tfc
import scipy.special


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def quantize(inputs, medians=None, method="SGAo-Q", **params):
    if method == "Identity":
        # (inputs - medians) + medians = inputs
        return inputs

    if method == "AUN-Q":
        half = tf.constant(0.5)
        noise = tf.random.uniform(tf.shape(inputs), -half, half)
        # (inputs - medians) + noise + medians = inputs + noise
        outputs = tf.math.add_n([inputs, noise])
        return outputs

    if medians is not None:
        inputs_ = inputs - medians
    else:
        inputs_ = inputs

    if method == "STE-Q":
        outputs = tf.round(inputs_)

        if medians is not None:
            outputs += medians
        outputs = tf.stop_gradient(outputs - inputs) + inputs

    elif method in {"St-Q", "SRA-Q"}:
        diff = inputs_ - tf.floor(inputs_)

        if method == "St-Q":
            probability = diff
        else:
            tau = params["tau"]
            likelihood_up = tf.exp(-tf.atanh(diff) / tau)
            likelihood_down = tf.exp(-tf.atanh(1 - diff) / tau)
            probability = likelihood_down / (likelihood_up + likelihood_down)
        delta = tf.cast(
            (probability >= tf.random.uniform(tf.shape(probability))),
            tf.float32,
        )
        outputs = tf.floor(inputs_) + delta
        if medians is not None:
            outputs += medians
        outputs = tf.stop_gradient(outputs - inputs) + inputs

    elif method == "U-Q":
        half = tf.constant(0.5)
        # random value, shape: (N, 1, 1, 1)
        noise = tf.random.uniform(tf.shape(inputs), -half, half)[
            :, 0:1, 0:1, 0:1
        ]
        outputs = tf.round(inputs_ + noise) - noise
        if medians is not None:
            outputs += medians
        outputs = tf.stop_gradient(outputs - inputs) + inputs
        # tf.round(inputs - medians + noise) - noise + medians
        # = tf.round(inputs + nmm) - nmm, nmm = noise minus medians
    elif method == "SGA-Q":
        tau = params["tau2"]
        # use Gumbel Softmax implemented in tfp.distributions.RelaxedOneHotCategorical
        # The code is modified from https://github.com/mandt-lab/improving-inference-for-neural-image-compression/blob/main/sga.py#L110-L121
        # copyright @yiboyang
        epsilon = 1e-5
        y_floor = tf.floor(inputs_)
        y_ceil = tf.ceil(inputs_)
        y_bds = tf.stack([y_floor, y_ceil], axis=-1)
        ry_logits = tf.stack(
            [
                -tf.math.atanh(
                    tf.clip_by_value(inputs_ - y_floor, -1 + epsilon, 1 - epsilon)
                )
                / tau,
                -tf.math.atanh(
                    tf.clip_by_value(y_ceil - inputs_, -1 + epsilon, 1 - epsilon)
                )
                / tau,
            ],
            axis=-1,
        )
        # last dim are logits for DOWN or UP
        ry_dist = tfp.distributions.RelaxedOneHotCategorical(tau, logits=ry_logits)
        ry_sample = ry_dist.sample()
        outputs = tf.reduce_sum(y_bds * ry_sample, axis=-1)
        if medians is not None:
            outputs += medians

    elif method == "sigmoid":
        T = params["T"]
        diff = inputs_ - tf.floor(inputs_)
        temp = diff * T
        # following the original implementation
        # https://github.com/aliyun/alibabacloud-quantization-networks/blob/master/anybit.py#L26
        temp = tf.clip_by_value(temp, -10.0, 10.0)
        outputs = tf.sigmoid(temp) + tf.floor(inputs_)
        if medians is not None:
            outputs += medians

    elif method == "DS-Q":
        k = params["k"]
        y_floor = tf.floor(inputs_)
        # 0 <= diff <= 1
        diff = inputs_ - y_floor
        # -1 <= phi <= 1
        phi = tf.tanh((diff - 0.5) * k) / tf.tanh(0.5 * k)
        # y_floor <= y_phi <= y_floor + 1
        y_phi = (1 + phi) / 2 + y_floor
        outputs = tf.round(inputs_)
        outputs = tf.stop_gradient(outputs - y_phi) + y_phi
        if medians is not None:
            outputs += medians
    else:
        raise NotImplementedError
    return outputs


class RoundingGaussianConditional(tfc.SymmetricConditional):
    # SRA-Q
    tau: float = 0.5
    # SGAo-Q
    tau2: float = 0.5
    # sigmoid
    T: float = 1.0
    # DS-Q, DSl-Q
    k: float = 1.0

    def __init__(self, scale, scale_table,
                scale_bound=None, mean=None, indexes=None, approx: str = "AUN-Q", sub_mean: bool = False, **kwargs):
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


class RoundingEntropyBottleneck(tfc.EntropyBottleneck):
    def __init__(
        self,
        init_scale=10,
        filters=(3, 3, 3),
        data_format="channels_last",
        approx="STE-Q",
        sub_mean: bool = False,
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
        if self.approx == "DSl-Q":
            raise NotImplementedError
            # self.quantizer = DSQ(self.k_init)

    def _quantize(self, inputs, mode):
        _, _, _, input_slices = self._get_input_dims()
        medians = self._medians[input_slices]

        if mode == "noise":
            if not self.sub_mean:
                medians = None
            if self.approx == "DSl-Q":
                assert medians is None
                self.quantizer(inputs)
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


def analysis_transform(tensor, num_filters, shallow: bool = False, heatmap: bool = False):
    """Builds the analysis transform."""
    latent_size: int = num_filters + 1 if heatmap else num_filters

    if shallow:
        with tf.variable_scope("analysis"):
            for i in range(4):
                with tf.variable_scope(f"layer_{i}"):
                    layer = tfc.SignalConv2D(
                        num_filters,
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
                        num_filters,
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
                        num_filters,
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
                        num_filters,
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
                        num_filters,
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
                        num_filters,
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
                    tensor = NonLocalAttentionBlock(tensor, num_filters, scope="NLAB_0")

            else:
                with tf.variable_scope("Block_" + str(i) + "_layer_2"):
                    layer = tfc.SignalConv2D(
                        latent_size,
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
                tensor = NonLocalAttentionBlock(tensor, latent_size, scope="NLAB_1")

        return tensor


def hyper_analysis(tensor, num_filters, shallow: bool = False):
    """Build the analysis transform in hyper"""
    if shallow:
        with tf.variable_scope("hyper_analysis"):
            for i in range(3):
                with tf.variable_scope(f"layer_{i}"):
                    layer = tfc.SignalConv2D(
                        num_filters,
                        (3, 3) if i == 0 else (5, 5),
                        corr=True,
                        strides_down=1 if i == 0 else 2,
                        padding="same_zeros",
                        use_bias=True if i < 2 else False,
                        activation=tf.nn.relu if i < 2 else None,
                    )
                    tensor = layer(tensor)
            return tensor

    with tf.variable_scope("hyper_analysis"):
        with tf.variable_scope("layer_0"):
            layer = tfc.SignalConv2D(
                num_filters,
                (3, 3),
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.leaky_relu,
                name="signal_conv2d",
            )
            tensor = layer(tensor)

        # One 5x5 is replaced by two 3x3 filters
        with tf.variable_scope("layer_1"):
            layer = tfc.SignalConv2D(
                num_filters,
                (3, 3),
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.leaky_relu,
                name="signal_conv2d",
            )
            tensor = layer(tensor)

        with tf.variable_scope("layer_2"):
            layer = tfc.SignalConv2D(
                num_filters,
                (3, 3),
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.leaky_relu,
                name="signal_conv2d",
            )
            tensor = layer(tensor)

        # One 5x5 is replaced by two 3x3 filters
        with tf.variable_scope("layer_3"):
            layer = tfc.SignalConv2D(
                num_filters,
                (3, 3),
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.leaky_relu,
                name="signal_conv2d",
            )
            tensor = layer(tensor)

        with tf.variable_scope("layer_4"):
            layer = tfc.SignalConv2D(
                num_filters,
                (3, 3),
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=None,
                name="signal_conv2d",
            )
            tensor = layer(tensor)

    return tensor


def synthesis_transform(tensor, num_filters, shallow: bool = False):
    """Builds the synthesis transform."""
    if shallow:
        with tf.variable_scope("synthesis"):
            for i in range(4):
                with tf.variable_scope(f"layer_{i}"):
                    layer = tfc.SignalConv2D(
                        num_filters if i < 3 else 3,
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
                tensor = NonLocalAttentionBlock(tensor, num_filters, scope="NLAB_0")

            if i == 2:
                # Add one NLAM
                tensor = NonLocalAttentionBlock(tensor, num_filters, scope="NLAB_1")

            with tf.variable_scope("Block_" + str(i) + "_layer_0"):
                layer = tfc.SignalConv2D(
                    num_filters,
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
                    num_filters,
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
                        num_filters * 4,
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
                        num_filters * 4,
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
                        num_filters,
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


def hyper_synthesis(tensor, num_filters, shallow: bool = False):
    """Builds the hyper synthesis transform"""
    if shallow:
        with tf.variable_scope("hyper_synthesis", reuse=tf.AUTO_REUSE):
            for i in range(3):
                with tf.variable_scope(f"layer_{i}"):
                    layer = tfc.SignalConv2D(
                        num_filters,
                        (5, 5) if i < 2 else (3, 3),
                        name="layer_0",
                        corr=False,
                        strides_up=2 if i < 2 else 1,
                        padding="same_zeros",
                        use_bias=True,
                        kernel_parameterizer=None,
                        activation=tf.nn.relu if i < 2 else None,
                    )
                    tensor = layer(tensor)
            return tensor

    with tf.variable_scope("hyper_synthesis", reuse=tf.AUTO_REUSE):
        # One 5x5 is replaced by two 3x3 filters
        with tf.variable_scope("layer_0"):
            layer = tfc.SignalConv2D(
                num_filters,
                (3, 3),
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.leaky_relu,
                name="signal_conv2d",
            )
            tensor = layer(tensor)

        with tf.variable_scope("layer_1"):
            layer = tfc.SignalConv2D(
                num_filters,
                (3, 3),
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.leaky_relu,
                name="signal_conv2d",
            )
            tensor = layer(tensor)

        # One 5x5 is replaced by two 3x3 filters
        with tf.variable_scope("layer_2"):
            layer = tfc.SignalConv2D(
                num_filters * 1.5,
                (3, 3),
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.leaky_relu,
                name="signal_conv2d",
            )
            tensor = layer(tensor)

        with tf.variable_scope("layer_3"):
            layer = tfc.SignalConv2D(
                num_filters * 1.5,
                (3, 3),
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.leaky_relu,
                name="signal_conv2d",
            )
            tensor = layer(tensor)

        with tf.variable_scope("layer_4"):
            layer = tfc.SignalConv2D(
                num_filters * 2,
                (3, 3),
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                activation=None,
                name="signal_conv2d",
            )
            tensor = layer(tensor)

        return tensor


def masked_conv2d(
    inputs,
    num_outputs,
    kernel_shape,  # [kernel_height, kernel_width]
    mask_type,  # None, "A" or "B",
    strides=[1, 1],  # [column_wise_stride, row_wise_stride]
    padding="SAME",
    activation_fn=None,
    weights_initializer=tf.contrib.layers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(),
    biases_regularizer=None,
    scope="masked",
):

    with tf.variable_scope(scope):
        mask_type = mask_type.lower()
        batch_size, height, width, channel = inputs.get_shape().as_list()

        kernel_h, kernel_w = kernel_shape
        stride_h, stride_w = strides

        assert (
            kernel_h % 2 == 1 and kernel_w % 2 == 1
        ), "kernel height and width should be odd number"

        center_h = kernel_h // 2
        center_w = kernel_w // 2

        weights_shape = [kernel_h, kernel_w, channel, num_outputs]
        weights = tf.get_variable(
            "weights",
            weights_shape,
            tf.float32,
            weights_initializer,
            weights_regularizer,
        )

        if mask_type is not None:
            mask = np.ones((kernel_h, kernel_w, channel, num_outputs), dtype=np.float32)

            mask[center_h, center_w + 1 :, :, :] = 0.0
            mask[center_h + 1 :, :, :, :] = 0.0

            if mask_type == "a":
                mask[center_h, center_w, :, :] = 0.0

            weights *= tf.constant(mask, dtype=tf.float32)
            tf.add_to_collection("conv2d_weights_%s" % mask_type, weights)

        outputs = tf.nn.conv2d(
            inputs, weights, [1, stride_h, stride_w, 1], padding=padding, name="outputs"
        )
        tf.add_to_collection("conv2d_outputs", outputs)

        if biases_initializer != None:
            biases = tf.get_variable(
                "biases",
                [
                    num_outputs,
                ],
                tf.float32,
                biases_initializer,
                biases_regularizer,
            )
            outputs = tf.nn.bias_add(outputs, biases, name="outputs_plus_b")

        if activation_fn:
            outputs = activation_fn(outputs, name="outputs_with_fn")

        return outputs


def entropy_parameter(tensor, inputs, num_filters, training, activation="noise", activation_ha=None, n_gmm: int = 3, **kwargs):
    """tensor: the output of hyper autoencoder (phi) to generate the mean and variance
    inputs: the variable needs to be encoded. (y)
    """
    assert n_gmm in {1, 3}
    with tf.variable_scope("entropy_parameter", reuse=tf.AUTO_REUSE):

        half = tf.constant(0.5)

        if training:
            values = quantize(inputs, method=activation)
            values_ha = quantize(inputs, method=activation_ha if activation_ha else activation, **kwargs)

        else:  # inference
            # if inputs is not None: #compress
            values = tf.round(inputs)
            values_ha = values

        masked = masked_conv2d(values_ha, num_filters * 2, [5, 5], "A", scope="masked")
        tensor = tf.concat([masked, tensor], axis=3)

        with tf.variable_scope("layer_0"):
            layer = tfc.SignalConv2D(
                640,
                (1, 1),
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.leaky_relu,
            )
            tensor = layer(tensor)

        with tf.variable_scope("layer_1"):
            layer = tfc.SignalConv2D(
                640,
                (1, 1),
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.leaky_relu,
            )
            tensor = layer(tensor)

        with tf.variable_scope("layer_2"):
            layer = tfc.SignalConv2D(
                num_filters * 3 * n_gmm,
                (1, 1),
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=False,
                activation=None,
            )
            tensor = layer(tensor)

        # =========Gaussian Mixture Model=========
        if n_gmm == 3:
            # (N, H, W, 9C)
            prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = tf.split(
                tensor, num_or_size_splits=9, axis=3
            )
            scale0 = tf.abs(scale0)
            scale1 = tf.abs(scale1)
            scale2 = tf.abs(scale2)

            probs = tf.stack([prob0, prob1, prob2], axis=-1)
            probs = tf.nn.softmax(probs, axis=-1)

            # To merge them together
            means = tf.stack([mean0, mean1, mean2], axis=-1)
            variances = tf.stack([scale0, scale1, scale2], axis=-1)

            # =======================================
            # Calculate the likelihoods for inputs
            dist_0 = tfd.Normal(loc=mean0, scale=scale0, name="dist_0")
            dist_1 = tfd.Normal(loc=mean1, scale=scale1, name="dist_1")
            dist_2 = tfd.Normal(loc=mean2, scale=scale2, name="dist_2")

            # =========Gaussian Mixture Model=========
            likelihoods_0 = dist_0.cdf(values + half) - dist_0.cdf(values - half)
            likelihoods_1 = dist_1.cdf(values + half) - dist_1.cdf(values - half)
            likelihoods_2 = dist_2.cdf(values + half) - dist_2.cdf(values - half)

            likelihoods = (
                probs[:, :, :, :, 0] * likelihoods_0
                + probs[:, :, :, :, 1] * likelihoods_1
                + probs[:, :, :, :, 2] * likelihoods_2
            )

            # =======REVISION: Robust version ==========
            edge_min = (
                probs[:, :, :, :, 0] * dist_0.cdf(values + half)
                + probs[:, :, :, :, 1] * dist_1.cdf(values + half)
                + probs[:, :, :, :, 2] * dist_2.cdf(values + half)
            )

            edge_max = (
                probs[:, :, :, :, 0] * (1.0 - dist_0.cdf(values - half))
                + probs[:, :, :, :, 1] * (1.0 - dist_1.cdf(values - half))
                + probs[:, :, :, :, 2] * (1.0 - dist_2.cdf(values - half))
            )
            likelihoods = tf.where(
                values < -254.5,
                edge_min,
                tf.where(values > 255.5, edge_max, likelihoods),
            )
        else:
            # (N, H, W, 3C) -> (N, H, W, C), (N, H, W, C), (N, H, W, C)
            prob0, mean0, scale0 = tf.split(tensor, num_or_size_splits=3, axis=3)
            scale0 = tf.abs(scale0)
            # (N, H, W, C, 1)
            probs = tf.stack([prob0], axis=-1)
            # NOTE: the probs is always 1.
            probs = tf.nn.softmax(probs, axis=-1)

            # To merge them together
            # (N, H, W, C, 1)
            means = tf.stack([mean0], axis=-1)
            variances = tf.stack([scale0], axis=-1)

            # =======================================
            # Calculate the likelihoods for inputs
            dist_0 = tfd.Normal(loc=mean0, scale=scale0, name="dist_0")

            # =========Gaussian Mixture Model=========
            likelihoods_0 = dist_0.cdf(values + half) - dist_0.cdf(values - half)

            likelihoods = probs[:, :, :, :, 0] * likelihoods_0

            # =======REVISION: Robust version ==========
            edge_min = probs[:, :, :, :, 0] * dist_0.cdf(values + half)
            edge_max = probs[:, :, :, :, 0] * (1.0 - dist_0.cdf(values - half))
            likelihoods = tf.where(
                values < -254.5,
                edge_min,
                tf.where(values > 255.5, edge_max, likelihoods),
            )

        likelihood_lower_bound = tf.constant(1e-6)
        likelihood_upper_bound = tf.constant(1.0)
        likelihoods = tf.minimum(
            tf.maximum(likelihoods, likelihood_lower_bound), likelihood_upper_bound
        )

    return values, likelihoods, means, variances, probs


def calc_pmf(samples, weight, sigma, mu, TINY, n_gmm: int = 3):
    pmf = 0.0
    for i in range(n_gmm):
        pmf += weight[i] * (
            0.5
            * (
                1
                + scipy.special.erf(
                    (samples + 0.5 - mu[i]) / ((sigma[i] + TINY) * 2 ** 0.5)
                )
            )
            - 0.5
            * (
                1
                + scipy.special.erf(
                    (samples - 0.5 - mu[i]) / ((sigma[i] + TINY) * 2 ** 0.5)
                )
            )
        )
    return pmf
