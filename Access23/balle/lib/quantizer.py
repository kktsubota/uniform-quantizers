import tensorflow.compat.v1 as tf

from lib.ops import quantize


class DSQ(tf.keras.layers.Layer):

    def __init__(self, k_init: float, **kwargs):
        self.k_init = k_init
        super(DSQ, self).__init__(**kwargs)

    def build(self, input_shape):
        initializer = tf.keras.initializers.Constant(value=self.k_init)
        self.k = self.add_weight(name="k", shape=(1,), initializer=initializer, trainable=True)
        super(DSQ, self).build(input_shape)

    def call(self, x):
        # does not work
        # k: np.ndarray = max(self.get_weights(), self.k_min)
        # self.set_weights([k])

        # self.k stops around self.k_min because self.k_min is used and k is no longer used when self.k < self.k_min.
        # k = tf.maximum(self.k, tf.constant(self.k_min))
        return quantize(x, method="DS-Q", k=self.k)

    def compute_output_shape(self, input_shape):
        return input_shape


class NonUQ(tf.keras.layers.Layer):

    def __init__(self, n_centers: int, sigma: float, **kwargs):
        self.n_centers = n_centers
        self.sigma = sigma
        super(NonUQ, self).__init__(**kwargs)

    def build(self, input_shape):
        initializer = tf.random_uniform_initializer(minval=-2, maxval=2, seed=666)
        self.centers = self.add_weight(name="centers", shape=(len(self.n_centers),), initializer=initializer, trainable=True)
        super(NonUQ, self).build(input_shape)

    def call(self, x):
        return quantize(x, method="NonUfix-Q", centers=self.centers, sigma=self.sigma)

    def compute_output_shape(self, input_shape):
        return input_shape
