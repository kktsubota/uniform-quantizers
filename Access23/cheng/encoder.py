#!/usr/bin/env python3
import argparse
from collections import defaultdict
from glob import glob
from multiprocessing import Pool
import os
from pathlib import Path
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_compression as tfc
from range_coder import RangeEncoder

from network import (
    calc_pmf,
    analysis_transform,
    hyper_analysis,
    hyper_synthesis,
    entropy_parameter,
    synthesis_transform,
    RoundingGaussianConditional,
    RoundingEntropyBottleneck,
)
from utils import load_image
from lib.ops import get_heatmap3D, mask_with_heatmap, transpose_NCHW_to_NHWC, transpose_NHWC_to_NCHW
from nonuq import NonUQEntropyModel, Config



SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def compress_factorized(
    input: str,
    output: str,
    num_filters: int,
    checkpoint_dir: Path,
    shallow: bool = False,
    heatmap: bool = False,
    n_gmm: int = 3,
    quantizer: str = "AUN-Q",
):
    tf.set_random_seed(1)
    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        # Load input image and add batch dimension.

        x = load_image(input)

        # Pad the x to x_pad
        mod = tf.constant([64, 64, 1], dtype=tf.int32)
        div = tf.ceil(tf.truediv(tf.shape(x), mod))
        div = tf.cast(div, tf.int32)
        paddings = tf.subtract(tf.multiply(div, mod), tf.shape(x))
        paddings = tf.expand_dims(paddings, 1)
        paddings = tf.concat(
            [tf.convert_to_tensor(np.zeros((3, 1)), dtype=tf.int32), paddings], axis=1
        )

        x_pad = tf.pad(x, paddings, "REFLECT")
        x = tf.expand_dims(x, 0)
        x.set_shape([1, None, None, 3])
        x_shape = tf.shape(x)

        x_pad = tf.expand_dims(x_pad, 0)
        x_pad.set_shape([1, None, None, 3])

        # Transform and compress the image, then remove batch dimension.
        y = analysis_transform(x_pad, num_filters, shallow, heatmap)
        if heatmap:
            y_nchw = transpose_NHWC_to_NCHW(y)
            heatmap3D = get_heatmap3D(y_nchw)
            y_masked = mask_with_heatmap(y_nchw, heatmap3D)
            y = transpose_NCHW_to_NHWC(y_masked)

        # Build a hyper autoencoder
        if quantizer == "NonU-Q":
            pc_config = Config()
            entropy_bottleneck = NonUQEntropyModel(pc_config)
        else:
            entropy_bottleneck = RoundingEntropyBottleneck()
            string = entropy_bottleneck.compress(y)

        y_hat, y_likelihoods = entropy_bottleneck(y, training=False)
        x_hat = synthesis_transform(y_hat, num_filters, shallow)
        x_hat = x_hat[:, : x_shape[1], : x_shape[2], :]

        num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

        # Total number of bits divided by number of pixels.
        eval_bpp = tf.reduce_sum(tf.log(y_likelihoods)) / (-np.log(2) * num_pixels)

        # Bring both images back to 0..255 range.
        x *= 255
        x_hat = tf.clip_by_value(x_hat, 0, 1)
        x_hat = tf.round(x_hat * 255)

        mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
        psnr = tf.image.psnr(x_hat, x, 255)
        msssim = tf.image.ssim_multiscale(x_hat, x, 255)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if checkpoint_dir is None:
            latest = "models/model-1399000"  # lambda = 14
        else:
            latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)

        print(latest)
        tf.train.Saver().restore(sess, save_path=latest)

        if quantizer != "NonU-Q":
            tensors = [
                string,
                tf.shape(x)[1:-1],
                tf.shape(y)[1:-1],
            ]
            arrays = sess.run(tensors)

            # Write a binary file with the shape information and the compressed string.
            packed = tfc.PackedTensors()
            packed.pack(tensors, arrays)
            with open(output, "wb") as f:
                f.write(packed.string)

        _, probability = entropy_bottleneck(y, training=False)

        # If requested, transform the quantized image back and measure performance.
        eval_bpp, mse, psnr, msssim, num_pixels, probability = sess.run(
            [eval_bpp, mse, psnr, msssim, num_pixels, probability]
        )

        if quantizer != "NonU-Q":
            # The actual bits per pixel including overhead.
            bpp = len(packed.string) * 8 / num_pixels
            bpp_estimated = -np.log2(probability).sum() / num_pixels
        else:
            bpp = 0.0
            bpp_estimated = probability.sum() / num_pixels

        print("Actual bits per pixel for this image: {:0.4}".format(bpp))
        print("Estimated bits per pixel for this image: {:0.4}".format(bpp_estimated))
        print("PSNR (dB) : {:0.4}".format(psnr[0]))
        print("MS-SSIM : {:0.4}".format(msssim[0]))
        return {
            "actual bpp": bpp,
            "bpp estimated": bpp_estimated,
            "side bpp": 0.0,
            "psnr": psnr[0],
            "ms-ssim": msssim[0],
        }


def compress_hyper(
    input: str,
    output: str,
    num_filters: int,
    checkpoint_dir: Path,
    shallow: bool = False,
    n_gmm: int = 3,
):
    tf.set_random_seed(1)
    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        # Load input image and add batch dimension.

        x = load_image(input)

        # Pad the x to x_pad
        mod = tf.constant([64, 64, 1], dtype=tf.int32)
        div = tf.ceil(tf.truediv(tf.shape(x), mod))
        div = tf.cast(div, tf.int32)
        paddings = tf.subtract(tf.multiply(div, mod), tf.shape(x))
        paddings = tf.expand_dims(paddings, 1)
        paddings = tf.concat(
            [tf.convert_to_tensor(np.zeros((3, 1)), dtype=tf.int32), paddings], axis=1
        )

        x_pad = tf.pad(x, paddings, "REFLECT")
        x = tf.expand_dims(x, 0)
        x.set_shape([1, None, None, 3])
        x_shape = tf.shape(x)

        x_pad = tf.expand_dims(x_pad, 0)
        x_pad.set_shape([1, None, None, 3])

        # Transform and compress the image, then remove batch dimension.
        y = analysis_transform(x_pad, num_filters, shallow)

        # Build a hyper autoencoder
        z = hyper_analysis(abs(y), num_filters, shallow)
        entropy_bottleneck = RoundingEntropyBottleneck()
        string = entropy_bottleneck.compress(z)
        string = tf.squeeze(string, axis=0)

        z_tilde, z_likelihoods = entropy_bottleneck(z, training=False)

        # To decompress the z_tilde back to avoid the inconsistence error
        string_rec = tf.expand_dims(string, 0)
        z_tilde = entropy_bottleneck.decompress(
            string_rec, tf.shape(z)[1:], channels=num_filters
        )
        sigma = hyper_synthesis(z_tilde, num_filters, shallow)

        scale_table = np.exp(
            np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS)
        )
        conditional_bottleneck = RoundingGaussianConditional(sigma, scale_table)
        side_string = entropy_bottleneck.compress(z)
        string = conditional_bottleneck.compress(y)

        y_hat, y_likelihoods = conditional_bottleneck(y, training=False)
        x_hat = synthesis_transform(y_hat, num_filters, shallow)
        x_hat = x_hat[:, : x_shape[1], : x_shape[2], :]

        num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

        # Total number of bits divided by number of pixels.
        eval_bpp = (
            tf.reduce_sum(tf.log(y_likelihoods)) + tf.reduce_sum(tf.log(z_likelihoods))
        ) / (-np.log(2) * num_pixels)

        # Bring both images back to 0..255 range.
        x *= 255
        x_hat = tf.clip_by_value(x_hat, 0, 1)
        x_hat = tf.round(x_hat * 255)

        mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
        psnr = tf.image.psnr(x_hat, x, 255)
        msssim = tf.image.ssim_multiscale(x_hat, x, 255)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if checkpoint_dir is None:
            latest = "models/model-1399000"  # lambda = 14
        else:
            latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)

        print(latest)
        tf.train.Saver().restore(sess, save_path=latest)

        tensors = [
            string,
            side_string,
            tf.shape(x)[1:-1],
            tf.shape(y)[1:-1],
            tf.shape(z)[1:-1],
        ]
        arrays = sess.run(tensors)

        # Write a binary file with the shape information and the compressed string.
        packed = tfc.PackedTensors()
        packed.pack(tensors, arrays)
        with open(output, "wb") as f:
            f.write(packed.string)

        # If requested, transform the quantized image back and measure performance.
        eval_bpp, mse, psnr, msssim, num_pixels = sess.run(
            [eval_bpp, mse, psnr, msssim, num_pixels]
        )

        # The actual bits per pixel including overhead.
        bpp = len(packed.string) * 8 / num_pixels

        print("Actual bits per pixel for this image: {:0.4}".format(bpp))
        print("PSNR (dB) : {:0.4}".format(psnr[0]))
        print("MS-SSIM : {:0.4}".format(msssim[0]))
        return {
            "actual bpp": bpp,
            "side bpp": 0.0,
            "psnr": psnr[0],
            "ms-ssim": msssim[0],
        }


def compress(
    input: str,
    output: str,
    num_filters: int,
    checkpoint_dir: Path,
    shallow: bool = False,
    n_gmm: int = 3,
):
    intermediate_dir = os.path.join(checkpoint_dir, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)
    print(intermediate_dir)
    start = time.time()
    tf.set_random_seed(1)
    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        # Load input image and add batch dimension.

        x = load_image(input)

        # Pad the x to x_pad
        mod = tf.constant([64, 64, 1], dtype=tf.int32)
        div = tf.ceil(tf.truediv(tf.shape(x), mod))
        div = tf.cast(div, tf.int32)
        paddings = tf.subtract(tf.multiply(div, mod), tf.shape(x))
        paddings = tf.expand_dims(paddings, 1)
        paddings = tf.concat(
            [tf.convert_to_tensor(np.zeros((3, 1)), dtype=tf.int32), paddings], axis=1
        )

        x_pad = tf.pad(x, paddings, "REFLECT")
        x = tf.expand_dims(x, 0)
        x.set_shape([1, None, None, 3])

        x_pad = tf.expand_dims(x_pad, 0)
        x_pad.set_shape([1, None, None, 3])

        # Transform and compress the image, then remove batch dimension.
        y = analysis_transform(x_pad, num_filters, shallow)

        # Build a hyper autoencoder
        z = hyper_analysis(y, num_filters, shallow)
        # entropy_bottleneck = tfc.EntropyBottleneck()
        entropy_bottleneck = RoundingEntropyBottleneck()
        string = entropy_bottleneck.compress(z)
        string = tf.squeeze(string, axis=0)

        z_tilde, z_likelihoods = entropy_bottleneck(z, training=False)

        # To decompress the z_tilde back to avoid the inconsistence error
        string_rec = tf.expand_dims(string, 0)
        z_tilde = entropy_bottleneck.decompress(
            string_rec, tf.shape(z)[1:], channels=num_filters
        )

        phi = hyper_synthesis(z_tilde, num_filters, shallow)

        # REVISION： for Gaussian Mixture Model (GMM), use window-based fast implementation
        # y = tf.clip_by_value(y, -255, 256)
        y_hat = tf.round(y)

        tiny_y = tf.placeholder(dtype=tf.float32, shape=[1] + [5] + [5] + [num_filters])
        tiny_phi = tf.placeholder(
            dtype=tf.float32, shape=[1, 5, 5, num_filters] if shallow else [1, 5, 5, num_filters * 2]
        )
        _, _, y_means, y_variances, y_probs = entropy_parameter(
            tiny_phi, tiny_y, num_filters, activation=None, training=False, n_gmm=n_gmm,
        )

        x_hat = synthesis_transform(y_hat, num_filters, shallow)

        num_pixels = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))
        x_hat = x_hat[0, : tf.shape(x)[1], : tf.shape(x)[2], :]

        # Mean squared error across pixels.
        x_hat = tf.clip_by_value(x_hat, 0, 1)
        x_hat = tf.round(x_hat * 255)
        difference = tf.squared_difference(x * 255, x_hat)
        mse = tf.reduce_mean(difference)

        with tf.Session() as sess:
            # print(tf.trainable_variables())
            sess.run(tf.global_variables_initializer())
            # Load the latest model checkpoint, get the compressed string and the tensor
            # shapes.
            if checkpoint_dir is None:
                latest = "models/model-1399000"  # lambda = 14
            else:
                latest = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)

            print(latest)
            tf.train.Saver().restore(sess, save_path=latest)

            string, x_shape, y_shape, num_pixels, y_hat_value, phi_value, diff_value = sess.run(
                [string, tf.shape(x), tf.shape(y), num_pixels, y_hat, phi, difference]
            )

            minmax = np.maximum(abs(y_hat_value.max()), abs(y_hat_value.min()))
            minmax = int(np.maximum(minmax, 1))
            # num_symbols = int(2 * minmax + 3)
            print(minmax)
            # print(num_symbols)

            # Fast implementations by only encoding non-zero channels with 128/8 = 16bytes overhead
            flag = np.zeros(y_shape[3], dtype=np.int)

            for ch_idx in range(y_shape[3]):
                if np.sum(abs(y_hat_value[:, :, :, ch_idx])) > 0:
                    flag[ch_idx] = 1

            (non_zero_idx,) = np.where(flag == 1)

            num = np.packbits(np.reshape(flag, [8, y_shape[3] // 8]))

            # ============== encode the bits for z===========
            if os.path.exists(output):
                os.remove(output)

            fileobj = open(output, mode="wb")
            fileobj.write(np.array(x_shape[1:-1], dtype=np.uint16).tobytes())
            fileobj.write(np.array([len(string), minmax], dtype=np.uint16).tobytes())
            fileobj.write(np.array(num, dtype=np.uint8).tobytes())
            fileobj.write(string)
            fileobj.close()

            # ============ encode the bits for y ==========
            print("INFO: start encoding y")
            encoder = RangeEncoder(output[:-4] + ".bin")
            samples = np.arange(0, minmax * 2 + 1)
            TINY = 1e-10

            kernel_size = 5
            pad_size = (kernel_size - 1) // 2

            padded_y = np.pad(
                y_hat_value,
                ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                "constant",
                constant_values=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            )
            padded_phi = np.pad(
                phi_value,
                ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                "constant",
                constant_values=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            )

            # (H, W, C)
            probability = np.ones((y_shape[1], y_shape[2], y_shape[3]), dtype=np.float32)
            for h_idx in range(y_shape[1]):
                for w_idx in range(y_shape[2]):

                    extracted_y = padded_y[
                        :, h_idx : h_idx + kernel_size, w_idx : w_idx + kernel_size, :
                    ]
                    extracted_phi = padded_phi[
                        :, h_idx : h_idx + kernel_size, w_idx : w_idx + kernel_size, :
                    ]

                    y_means_values, y_variances_values, y_probs_values = sess.run(
                        [y_means, y_variances, y_probs],
                        feed_dict={tiny_y: extracted_y, tiny_phi: extracted_phi},
                    )

                    for i in range(len(non_zero_idx)):
                        ch_idx = non_zero_idx[i]

                        mu = y_means_values[0, pad_size, pad_size, ch_idx, :] + minmax
                        sigma = y_variances_values[0, pad_size, pad_size, ch_idx, :]
                        weight = y_probs_values[0, pad_size, pad_size, ch_idx, :]

                        # Calculate the pmf/cdf
                        pmf = calc_pmf(samples, weight, sigma, mu, TINY, n_gmm=n_gmm)

                        # To avoid the zero-probability
                        pmf_clip = np.clip(pmf, 1.0 / 65536, 1.0)
                        pmf_clip = np.round(pmf_clip / np.sum(pmf_clip) * 65536)
                        cdf = list(np.add.accumulate(pmf_clip))
                        cdf = [0] + [int(i) for i in cdf]

                        symbol = np.int(y_hat_value[0, h_idx, w_idx, ch_idx] + minmax)

                        p = pmf_clip[symbol] / pmf_clip.sum()
                        probability[h_idx, w_idx, ch_idx] = p
                        encoder.encode([symbol], cdf)

            encoder.close()

            size_real = os.path.getsize(output) + os.path.getsize(output[:-4] + ".bin")
            bpp_real = size_real * 8 / num_pixels
            bpp_side = (os.path.getsize(output)) * 8 / num_pixels
            # -np.log2(1) = 0 bits for zero channels
            bpp_estimated = -np.log2(probability).sum() / num_pixels
            end = time.time()
            print("Time : {:0.3f}".format(end - start))

            psnr = sess.run(tf.image.psnr(x_hat, x * 255, 255))
            msssim = sess.run(tf.image.ssim_multiscale(x_hat, x * 255, 255))

            print("Actual bits per pixel for this image: {:0.4}".format(bpp_real))
            print("Estimated bits per pixel for this image (main): {:0.4}".format(bpp_estimated))
            print("Side bits per pixel for z: {:0.4}".format(bpp_side))
            print("PSNR (dB) : {:0.4}".format(psnr[0]))
            print("MS-SSIM : {:0.4}".format(msssim[0]))

            intermediate = {"probability (main)": probability, "difference": diff_value}
            np.save(os.path.join(intermediate_dir, os.path.basename(input) + ".npy"), intermediate)
            return {
                "actual bpp": bpp_real,
                "estimated main bpp": bpp_estimated,
                "side bpp": bpp_side,
                "psnr": psnr[0],
                "ms-ssim": msssim[0],
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_filters", type=int, default=128)
    parser.add_argument("--shallow", action="store_true")
    parser.add_argument("--n-gmm", type=int, default=3)
    parser.add_argument("--prior", type=str, default="context")
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--quantizer", default="AUN-Q")
    parser.add_argument("--dataset", choices={"kodak", "clic"})
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--input", default=None, type=Path)
    parser.add_argument("--out", default=None, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    default_root = {
        "kodak": "/home/tsubota/data/compression/Kodak/images",
        "clic": "/home/tsubota/data/compression/CLIC20/valid",
    }
    if args.input is None:
        args.input = Path(default_root[args.dataset])

    if args.out is None:
        args.out = Path(args.checkpoint_dir) / "compressed"

    args.out.mkdir(parents=True, exist_ok=True)
    image_files = list(args.input.glob("*.png"))

    arguments = [
        (
            image_file.as_posix(),
            (args.out / (image_file.name + ".npz")).as_posix(),
            args.num_filters,
            args.checkpoint_dir,
            args.shallow,
            args.heatmap,
            args.n_gmm,
            args.quantizer,
        )
        for image_file in image_files
    ]
    # DEBUG
    # score = compress(*arguments[0])
    # score = compress_factorized(*arguments[0])
    # return
    with Pool(args.workers) as p:
        if args.prior == "hyper":
            scores = p.starmap(compress_hyper, arguments)
        elif args.prior == "factorized":
            scores = p.starmap(compress_factorized, arguments)
        else:
            scores = p.starmap(compress, arguments)

    # scores = [compress(*arg) for arg in arguments]
    score_dict: Dict[str, List[float]] = defaultdict(list)

    keys_metric = scores[0].keys()
    for score in scores:
        for key in keys_metric:
            score_dict[key].append(score[key])

    df = pd.DataFrame.from_dict(score_dict)
    df.index = image_files
    if args.checkpoint_dir is None:
        args.checkpoint_dir = ""
    df.to_csv(os.path.join(args.checkpoint_dir, "{}.csv".format(args.dataset)))
    print(df.mean())


if __name__ == "__main__":
    main()
