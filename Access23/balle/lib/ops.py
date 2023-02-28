import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


# https://github.com/fab-jul/fjcommon/blob/master/fjcommon/tf_helpers.py
# MIT license (c) fab-jul
def transpose_NCHW_to_NHWC(t):
    return tf.transpose(t, (0, 2, 3, 1), name='to_NHWC')

# https://github.com/fab-jul/fjcommon/blob/master/fjcommon/tf_helpers.py
def transpose_NHWC_to_NCHW(t):
    return tf.transpose(t, (0, 3, 1, 2), name='to_NCHW')

# https://github.com/fab-jul/imgcomp-cvpr/blob/master/code/quantizer.py
# GPL-3.0 license (c) fab-jul
def phi_times_centers(phi, centers):
    matmul_innerproduct = phi * centers  # (B, C, m, L)
    return tf.reduce_sum(matmul_innerproduct, axis=3)  # (B, C, m)

# https://github.com/fab-jul/imgcomp-cvpr/blob/master/code/quantizer.py
def _quantize1d(x, centers, sigma: float, data_format: str):
    """
    :return: (softout, hardout, symbols_vol)
        each of same shape as x, softout, hardout will be float32, symbols_vol will be int64
    
        data_format == "NHWC": x is (N, H, W, C). returns (N, H, W, C).
            The input tensor is reshaped into (N, C, H, W) and processed internally.

        data_format == "NCHW": x is (N, C, H, W). returns (N, C, H, W).
    """
    _HARD_SIGMA = 1e7

    assert tf.float32.is_compatible_with(x.dtype), 'x should be float32'
    assert tf.float32.is_compatible_with(centers.dtype), 'centers should be float32'
    assert len(x.get_shape()) == 4, 'x should be NCHW or NHWC, got {}'.format(x.get_shape())
    assert len(centers.get_shape()) == 1, 'centers should be (L,), got {}'.format(centers.get_shape())

    if data_format == 'NHWC':
        x_t = transpose_NHWC_to_NCHW(x)
        softout, hardout, symbols_hard = _quantize1d(x_t, centers, sigma, data_format='NCHW')
        return tuple(map(transpose_NCHW_to_NHWC, (softout, hardout, symbols_hard)))

    # Note: from here on down, x is NCHW ---

    # count centers
    num_centers = centers.get_shape().as_list()[-1]

    with tf.name_scope('reshape_BCm1'):
        # reshape (B, C, w, h) to (B, C, m=w*h)
        x_shape_BCwh = tf.shape(x)
        B = x_shape_BCwh[0]  # B is not necessarily static
        C = int(x.shape[1])  # C is static
        x = tf.reshape(x, [B, C, -1])

        # make x into (B, C, m, 1)
        x = tf.expand_dims(x, axis=-1)

    with tf.name_scope('dist'):
        # dist is (B, C, m, L), contains | x_i - c_j | ^ 2
        dist = tf.square(tf.abs(x - centers))

    with tf.name_scope('phi_soft'):
        # (B, C, m, L)
        phi_soft = tf.nn.softmax(-sigma       * dist, dim=-1)
    with tf.name_scope('phi_hard'):
        # (B, C, m, L) probably not necessary due to the argmax!
        phi_hard = tf.nn.softmax(-_HARD_SIGMA * dist, dim=-1)

        symbols_hard = tf.argmax(phi_hard, axis=-1)
        phi_hard = tf.one_hot(symbols_hard, depth=num_centers, axis=-1, dtype=tf.float32)

    with tf.name_scope('softout'):
        softout = phi_times_centers(phi_soft, centers)
    with tf.name_scope('hardout'):
        hardout = phi_times_centers(phi_hard, centers)

    def reshape_to_BCwh(t_):
        with tf.name_scope('reshape_BCwh'):
            return tf.reshape(t_, x_shape_BCwh)
    return tuple(map(reshape_to_BCwh, (softout, hardout, symbols_hard)))


def quantize(inputs, medians=None, method="SGA-Q", **params):
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
        # MIT license (c) yiboyang
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
        # Apache-2.0 (c) jasonustc
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
    
    elif method == "NonUfix-Q":
        sigma = params["sigma"]
        centers = params["centers"]
        outputs = _quantize1d(inputs_, centers, sigma, data_format="NHWC")

        outputs = tf.stop_gradient(outputs - inputs_) + inputs_
        if medians is not None:
            outputs += medians

    else:
        raise NotImplementedError
    return outputs

# https://github.com/fab-jul/imgcomp-cvpr/blob/master/code/bits.py
# GPL-3.0 license (c) fab-jul
def bitcost_to_bpp(bit_cost, input_batch):
    """
    :param bit_cost: NChw
    :param input_batch: N3HW
    :return: Chw / HW, i.e., num_bits / num_pixels
    """
    assert bit_cost.shape.ndims == input_batch.shape.ndims == 4, 'Expected NChw and N3HW, got {} and {}'.format(
        bit_cost, input_batch)
    with tf.name_scope('bitcost_to_bpp'):
        num_bits = tf.reduce_sum(bit_cost, name='num_bits')
        return num_bits / tf.to_float(num_pixels_in_input_batch(input_batch))


# https://github.com/fab-jul/imgcomp-cvpr/blob/master/code/bits.py
def num_pixels_in_input_batch(input_batch):
    assert int(input_batch.shape[1]) == 3, 'Expected N3HW, got {}'.format(input_batch)
    with tf.name_scope('num_pixels'):
        return tf.reduce_prod(tf.shape(input_batch)) / 3


# https://github.com/fab-jul/imgcomp-cvpr/blob/master/code/autoencoder.py#L172-L200
# GPL-3.0 license (c) fab-jul
def get_heatmap3D(bottleneck):
    """
    create heatmap3D, where
        heatmap3D[x, y, c] = heatmap[x, y] - c \intersect [0, 1]
        assume the shape is (N, C, H, W)
    """
    assert bottleneck.shape.ndims == 4, bottleneck.shape

    with tf.name_scope('heatmap'):
        C = int(bottleneck.shape[1]) - 1  # -1 because first channel is heatmap

        heatmap_channel = bottleneck[:, 0, :, :]  # NHW
        heatmap2D = tf.nn.sigmoid(heatmap_channel) * C  # NHW
        c = tf.range(C, dtype=tf.float32)  # C

        # reshape heatmap2D for broadcasting
        heatmap = tf.expand_dims(heatmap2D, 1)  # N1HW
        # reshape c for broadcasting
        c = tf.reshape(c, (C, 1, 1))  # C11

        # construct heatmap3D
        # if heatmap[x, y] == C, then heatmap[x, y, c] == 1 \forall c \in {0, ..., C-1}
        heatmap3D = tf.maximum(tf.minimum(heatmap - c, 1), 0, name='heatmap3D')  # NCHW
        return heatmap3D


def mask_with_heatmap(bottleneck, heatmap3D):
    """Compute bottleneck with heatmap3D
    assume the shape is (N, C, H, W)
    Args:
        bottleneck (N, C + 1, H, W): latent. +1 denotes the heatmap channel
        heatmap3D (N, C, H, W): heatmap3D obtained by get_heatmap3D
    """
    with tf.name_scope('heatmap_mask'):
        bottleneck_without_heatmap = bottleneck[:, 1:, ...]
        return heatmap3D * bottleneck_without_heatmap
