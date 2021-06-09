import tensorflow as tf
import tensorflow_compression as tfc


class RoundingEntropyBottleneck(tfc.EntropyBottleneck):
    def __init__(
        self,
        init_scale=10,
        filters=(3, 3, 3),
        data_format="channels_last",
        activation="deterministic",
        **kwargs
    ):
        super(RoundingEntropyBottleneck, self).__init__(
            init_scale=init_scale, filters=filters, data_format=data_format, **kwargs
        )
        self.activation = activation
        self.tau = 0.5

    def _quantize(self, inputs, mode):
        # Add noise or quantize (and optionally dequantize in one step).
        half = tf.constant(0.5, dtype=self.dtype)
        _, _, _, input_slices = self._get_input_dims()

        medians = self._medians[input_slices]
        outputs = tf.math.floor(inputs + (half - medians))
        outputs = tf.cast(outputs, self.dtype)

        if mode == "noise":
            if self.activation == "deterministic":
                return tf.stop_gradient(outputs + medians - inputs) + inputs
            elif self.activation in {"stochastic", "sga"}:
                diff = (inputs - medians) - tf.floor(inputs - medians)
                if self.activation == "stochastic":
                    probability = diff
                else:
                    likelihood_up = tf.exp(-tf.atanh(diff) / self.tau)
                    likelihood_down = tf.exp(-tf.atanh(1 - diff) / self.tau)
                    probability = likelihood_down / (likelihood_up + likelihood_down)
                delta = tf.cast(
                    (probability >= tf.random.uniform(tf.shape(probability))),
                    tf.float32,
                )
                outputs = tf.floor(inputs - medians) + delta
                return tf.stop_gradient(outputs + medians - inputs) + inputs
            elif self.activation == "universal":
                # random value, shape: (N, 1, 1, 1)
                noise = tf.random.uniform(tf.shape(inputs), -half, half)[
                    :, 0:1, 0:1, 0:1
                ]
                outputs = tf.round(inputs + noise) - noise
                return tf.stop_gradient(outputs - inputs) + inputs
            else:
                raise NotImplementedError
        elif mode == "dequantize":
            return outputs + medians
        else:
            assert mode == "symbols", mode
            outputs = tf.cast(outputs, tf.int32)
            return outputs
