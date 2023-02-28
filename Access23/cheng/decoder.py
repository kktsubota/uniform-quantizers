#!/usr/bin/env python3
import argparse
import math
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
from range_coder import RangeDecoder

from network import (
    calc_pmf,
    hyper_synthesis,
    entropy_parameter,
    synthesis_transform,
    RoundingEntropyBottleneck,
)
from utils import save_image


def decompress(input, output, num_filters, checkpoint_dir):
    """Decompresses an image by a fast implementation."""

    start = time.time()

    tf.set_random_seed(1)
    tf.reset_default_graph()

    with tf.device("/cpu:0"):

        print(input)

        # Read the shape information and compressed string from the binary file.
        fileobj = open(input, mode="rb")
        x_shape = np.frombuffer(fileobj.read(4), dtype=np.uint16)
        length, minmax = np.frombuffer(fileobj.read(4), dtype=np.uint16)
        num = np.frombuffer(fileobj.read(16), dtype=np.uint8)
        string = fileobj.read(length)

        fileobj.close()

        flag = np.unpackbits(num)
        non_zero_idx = np.squeeze(np.where(flag == 1))

        # Get x_pad_shape, y_shape, z_shape
        pad_size = 64
        x_pad_shape = (
            [1]
            + [int(math.ceil(x_shape[0] / pad_size) * pad_size)]
            + [int(math.ceil(x_shape[1] / pad_size) * pad_size)]
            + [3]
        )
        y_shape = [1] + [x_pad_shape[1] // 16] + [x_pad_shape[2] // 16] + [num_filters]
        z_shape = [y_shape[1] // 4] + [y_shape[2] // 4] + [num_filters]

        # Add a batch dimension, then decompress and transform the image back.
        strings = tf.expand_dims(string, 0)

        entropy_bottleneck = RoundingEntropyBottleneck(dtype=tf.float32)
        z_tilde = entropy_bottleneck.decompress(strings, z_shape, channels=num_filters)
        phi = hyper_synthesis(z_tilde, num_filters)

        # Transform the quantized image back (if requested).
        tiny_y = tf.placeholder(dtype=tf.float32, shape=[1] + [5] + [5] + [num_filters])
        tiny_phi = tf.placeholder(
            dtype=tf.float32, shape=[1] + [5] + [5] + [num_filters * 2]
        )
        _, _, means, variances, probs = entropy_parameter(
            tiny_phi, tiny_y, num_filters, training=False
        )

        # Decode the x_hat usign the decoded y
        y_hat = tf.placeholder(dtype=tf.float32, shape=y_shape)
        x_hat = synthesis_transform(y_hat, num_filters)

        # Remove batch dimension, and crop away any extraneous padding on the bottom or right boundaries.
        x_hat = x_hat[0, : int(x_shape[0]), : int(x_shape[1]), :]

        # Write reconstructed image out as a PNG file.
        op = save_image(output, x_hat)

        # Load the latest model checkpoint, and perform the above actions.
        with tf.Session() as sess:
            latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
            tf.train.Saver().restore(sess, save_path=latest)

            phi_value = sess.run(phi)

            print("INFO: start decoding y")
            print(time.time() - start)

            decoder = RangeDecoder(input[:-4] + ".bin")
            samples = np.arange(0, minmax * 2 + 1)
            TINY = 1e-10

            # Fast implementation to decode the y_hat
            kernel_size = 5
            pad_size = (kernel_size - 1) // 2

            decoded_y = np.zeros(
                [1]
                + [y_shape[1] + kernel_size - 1]
                + [y_shape[2] + kernel_size - 1]
                + [num_filters]
            )
            padded_phi = np.pad(
                phi_value,
                ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                "constant",
                constant_values=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            )

            for h_idx in range(y_shape[1]):
                for w_idx in range(y_shape[2]):

                    y_means, y_variances, y_probs = sess.run(
                        [means, variances, probs],
                        feed_dict={
                            tiny_y: decoded_y[
                                :,
                                h_idx : h_idx + kernel_size,
                                w_idx : w_idx + kernel_size,
                                :,
                            ],
                            tiny_phi: padded_phi[
                                :,
                                h_idx : h_idx + kernel_size,
                                w_idx : w_idx + kernel_size,
                                :,
                            ],
                        },
                    )

                    for i in range(len(non_zero_idx)):
                        ch_idx = non_zero_idx[i]

                        mu = y_means[0, pad_size, pad_size, ch_idx, :] + minmax
                        sigma = y_variances[0, pad_size, pad_size, ch_idx, :]
                        weight = y_probs[0, pad_size, pad_size, ch_idx, :]
                        pmf = calc_pmf(samples, weight, sigma, mu, TINY)
                        pmf_clip = np.clip(pmf, 1.0 / 65536, 1.0)
                        pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                        cdf = list(np.add.accumulate(pmf_clip))
                        cdf = [0] + [int(i) for i in cdf]

                        decoded_y[0, h_idx + pad_size, w_idx + pad_size, ch_idx] = (
                            decoder.decode(1, cdf)[0] - minmax
                        )

            decoded_y = decoded_y[
                :, pad_size : y_shape[1] + pad_size, pad_size : y_shape[2] + pad_size, :
            ]

            sess.run(op, feed_dict={y_hat: decoded_y})

            end = time.time()
            print("Time (s): {:0.3f}".format(end - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_filters", type=int, default=128)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--input", default="images/kodim01.npz")
    args = parser.parse_args()

    output = args.input + ".png"
    decompress(args.input, output, args.num_filters, args.checkpoint_dir)


if __name__ == "__main__":
    main()
