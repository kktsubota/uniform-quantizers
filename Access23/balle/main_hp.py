import argparse
import os
import sys

from absl import app
from absl.flags import argparse_flags
import numpy as np
from lib.decoder import FourConvDecoder
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

from common import add_test_args, prepare_dataset, read_png, write_png
from lib.encoder import FourConvEncoder
from lib.entropy_model import RoundingEntropyBottleneck, RoundingGaussianConditional, NonUQEntropyModel, Config
from lib.ops import quantize


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


class HyperAnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform for the entropy model parameters."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(HyperAnalysisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters,
                (3, 3),
                name="layer_0",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.relu,
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_1",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.relu,
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_2",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=False,
                activation=None,
            ),
        ]
        super(HyperAnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class HyperSynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform for the entropy model parameters."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(HyperSynthesisTransform, self).__init__(*args, **kwargs)

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
                kernel_parameterizer=None,
                activation=tf.nn.relu,
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_1",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                kernel_parameterizer=None,
                activation=tf.nn.relu,
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (3, 3),
                name="layer_2",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                kernel_parameterizer=None,
                activation=None,
            ),
        ]
        super(HyperSynthesisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


def train(args):
    """Trains the model."""

    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    # Create input data pipeline.
    with tf.device("/cpu:0"):
        train_dataset = prepare_dataset(args)

    num_pixels = args.batchsize * args.patchsize ** 2

    # Get training patch from dataset.
    x = train_dataset.make_one_shot_iterator().get_next()

    # Instantiate model.
    analysis_transform = FourConvEncoder(args.num_filters)
    synthesis_transform = FourConvDecoder(args.num_filters)
    if args.qua_ent == "NonU-Q":
        pc_config = Config(args)
        entropy_bottleneck = NonUQEntropyModel(pc_config)
    else:
        hyper_analysis_transform = HyperAnalysisTransform(args.num_filters)
        hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters)
        entropy_bottleneck = RoundingEntropyBottleneck(approx=args.qua_ent, sub_mean=False)

    # tau scheduler
    step = tf.train.create_global_step()
    decaying_iter = tf.cast(step - args.tau_decay_iteration, tf.float32)
    # if decaying_iter < 0, tau should be 0.5.
    tau = tf.minimum(0.5, 0.5 * tf.exp(-args.tau_decay_factor * decaying_iter))

    # tau2 scheduler (same with tau)
    decaying_iter2 = tf.cast(step - args.tau2_decay_iteration, tf.float32)
    # if decaying_iter < 0, tau should be 0.5.
    tau2 = tf.minimum(0.5, 0.5 * tf.exp(-args.tau2_decay_factor * decaying_iter2))

    k_decay_iter = tf.cast(step - args.k_iteration, tf.float32)
    k = args.k_init + tf.maximum(0.0, (args.k_end - args.k_init) * k_decay_iter / (args.last_step - args.k_iteration))

    if args.qua_ent == "SRA-Q":
        entropy_bottleneck.tau = tau
    elif args.qua_ent == "SGA-Q":
        entropy_bottleneck.tau2 = tau2
    elif args.qua_ent in {"DSl-Q", "DS-Q"}:
        entropy_bottleneck.k = k

    # Build autoencoder and hyperprior.
    y = analysis_transform(x)
    if args.qua_ent == "NonU-Q":
        y_tilde, y_likelihoods = entropy_bottleneck(y, training=True)
        x_tilde = synthesis_transform(y_tilde)

        # Total number of bits divided by number of pixels.
        train_bpp = tf.reduce_sum(y_likelihoods) / num_pixels

    else:
        z = hyper_analysis_transform(abs(y))

        scale_table = np.exp(
            np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS)
        )

        # obtain y_dec and y_tilde
        if args.fix_qua:
            # the encoder and the hyper-encoder are fixed by actual quantization.
            z_tilde, z_likelihoods = entropy_bottleneck(z, training=False)
            sigma = hyper_synthesis_transform(z_tilde)
            conditional_bottleneck = RoundingGaussianConditional(sigma, scale_table, approx=args.qua_ent, sub_mean=False)
            y_tilde, y_likelihoods = conditional_bottleneck(y, training=False)
            x_tilde = synthesis_transform(y_tilde)

        else:
            z_tilde, z_likelihoods = entropy_bottleneck(z, training=True)
            if args.qua_dec == args.qua_ent:
                z_dec = z_tilde
            else:
                z_dec = quantize(z, None, method=args.qua_dec, tau=tau, tau2=tau2, k=k)
            sigma = hyper_synthesis_transform(z_dec)
            scale_table = np.exp(
                np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS)
            )
            conditional_bottleneck = RoundingGaussianConditional(sigma, scale_table, approx=args.qua_ent, sub_mean=False)
            if args.qua_ent == "SRA-Q":
                conditional_bottleneck.tau = tau
            elif args.qua_ent == "SGA-Q":
                conditional_bottleneck.tau2 = tau2
            elif args.qua_ent in {"DSl-Q", "DS-Q"}:
                conditional_bottleneck.k = k
            y_tilde, y_likelihoods = conditional_bottleneck(y, training=True)
            if args.qua_dec == args.qua_ent:
                y_dec = y_tilde
            else:
                y_dec = quantize(y, None, method=args.qua_dec, tau=tau, tau2=tau2, k=k)

            x_tilde = synthesis_transform(y_dec)

        # Total number of bits divided by number of pixels.
        train_bpp = (
            tf.reduce_sum(tf.log(y_likelihoods)) + tf.reduce_sum(tf.log(z_likelihoods))
        ) / (-np.log(2) * num_pixels)

    if args.mse_to_255:
        train_mse = tf.reduce_mean(tf.squared_difference(x * 255, x_tilde * 255))
    else:
        # Mean squared error across pixels.
        train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
        # Multiply by 255^2 to correct for rescaling.
        train_mse *= 255 ** 2

    # The rate-distortion cost.
    train_loss = args.lmbda * train_mse + train_bpp

    # add regularization of centers
    if args.qua_ent == "NonU-Q":
        reg = tf.to_float(pc_config.regularization_factor_centers)
        centers_reg = tf.identity(reg * tf.nn.l2_loss(entropy_bottleneck.centers), name='l2_reg')
        train_loss += centers_reg

    if args.reduce_lr:
        decay_steps = args.last_step - 80000
        adjusted_lr = tf.train.exponential_decay(
            1e-4,
            step,
            decay_steps,
            # decay factor
            0.1,
            staircase=True,
        )
        main_optimizer = tf.train.AdamOptimizer(learning_rate=adjusted_lr)

    else:
        # Minimize loss and auxiliary loss, and execute update op.
        main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    main_step = main_optimizer.minimize(train_loss, global_step=step)

    if args.adjust_aux_lr:
        aux_optimizer = tf.train.AdamOptimizer(learning_rate=adjusted_lr)
    else:
        aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    
    if args.qua_ent == "NonU-Q":
        train_op = tf.group(main_step)
    else:
        aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])
        train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

    tf.summary.scalar("loss", train_loss)
    tf.summary.scalar("bpp", train_bpp)
    tf.summary.scalar("mse", train_mse)

    hooks = [
        tf.train.StopAtStepHook(last_step=args.last_step),
        tf.train.NanTensorHook(train_loss),
    ]
    with tf.train.MonitoredTrainingSession(
        hooks=hooks,
        checkpoint_dir=args.checkpoint_dir,
        save_checkpoint_secs=3000,
        save_summaries_secs=600,
    ) as sess:
        while not sess.should_stop():
            sess.run(train_op)


def compress(args):
    """Compresses an image."""

    # Load input image and add batch dimension.
    x = read_png(args.input_file)
    x = tf.expand_dims(x, 0)
    x.set_shape([1, None, None, 3])
    x_shape = tf.shape(x)

    # Instantiate model.
    analysis_transform = FourConvEncoder(args.num_filters)
    synthesis_transform = FourConvDecoder(args.num_filters)

    if args.qua_ent == "NonU-Q":
        pc_config = Config(args)
        entropy_bottleneck = NonUQEntropyModel(pc_config)
    else:
        hyper_analysis_transform = HyperAnalysisTransform(args.num_filters)
        hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters)
        entropy_bottleneck = RoundingEntropyBottleneck()

    # Transform and compress the image.
    y = analysis_transform(x)

    if args.qua_ent == "NonU-Q":
        string = entropy_bottleneck.compress(y)

        # Transform the quantized image back (if requested).
        y_hat, y_likelihoods = entropy_bottleneck(y, training=False)
    else:
        y_shape = tf.shape(y)
        z = hyper_analysis_transform(abs(y))
        z_hat, z_likelihoods = entropy_bottleneck(z, training=False)
        sigma = hyper_synthesis_transform(z_hat)
        sigma = sigma[:, : y_shape[1], : y_shape[2], :]
        scale_table = np.exp(
            np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS)
        )
        conditional_bottleneck = RoundingGaussianConditional(sigma, scale_table)
        side_string = entropy_bottleneck.compress(z)
        string = conditional_bottleneck.compress(y)

        # Transform the quantized image back (if requested).
        y_hat, y_likelihoods = conditional_bottleneck(y, training=False)

    x_hat = synthesis_transform(y_hat)
    x_hat = x_hat[:, : x_shape[1], : x_shape[2], :]

    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

    # Total number of bits divided by number of pixels.
    if args.qua_ent == "NonU-Q":
        eval_bpp = (
            tf.reduce_sum(tf.log(y_likelihoods))
        ) / (-np.log(2) * num_pixels)
    else:
        eval_bpp = (
            tf.reduce_sum(tf.log(y_likelihoods)) + tf.reduce_sum(tf.log(z_likelihoods))
        ) / (-np.log(2) * num_pixels)

    # Bring both images back to 0..255 range.
    x *= 255
    x_hat = tf.clip_by_value(x_hat, 0, 1)
    x_hat = tf.round(x_hat * 255)

    mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
    psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
    msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))

    with tf.Session() as sess:
        # Load the latest model checkpoint, get the compressed string and the tensor
        # shapes.
        latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
        tf.train.Saver().restore(sess, save_path=latest)
        if args.qua_ent == "NonU-Q":
            tensors = [
                string,
                tf.shape(x)[1:-1],
                tf.shape(y)[1:-1],
            ]
        else:
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
        with open(args.output_file, "wb") as f:
            f.write(packed.string)

        # If requested, transform the quantized image back and measure performance.
        if args.verbose:
            eval_bpp, mse, psnr, msssim, num_pixels = sess.run(
                [eval_bpp, mse, psnr, msssim, num_pixels]
            )

            # The actual bits per pixel including overhead.
            bpp = len(packed.string) * 8 / num_pixels

            print("Mean squared error: {}".format(mse))
            print("PSNR (dB): {}".format(psnr))
            print("Multiscale SSIM: {}".format(msssim))
            print("Multiscale SSIM (dB): {}".format(-10 * np.log10(1 - msssim)))
            print("Information content in bpp: {}".format(eval_bpp))
            print("Actual bits per pixel: {}".format(bpp))


def decompress(args):
    """Decompresses an image."""

    # Read the shape information and compressed string from the binary file.
    string = tf.placeholder(tf.string, [1])
    side_string = tf.placeholder(tf.string, [1])
    x_shape = tf.placeholder(tf.int32, [2])
    y_shape = tf.placeholder(tf.int32, [2])
    z_shape = tf.placeholder(tf.int32, [2])
    with open(args.input_file, "rb") as f:
        packed = tfc.PackedTensors(f.read())
    tensors = [string, side_string, x_shape, y_shape, z_shape]
    arrays = packed.unpack(tensors)

    # Instantiate model.
    synthesis_transform = FourConvDecoder(args.num_filters)
    hyper_synthesis_transform = HyperSynthesisTransform(args.num_filters)
    entropy_bottleneck = RoundingEntropyBottleneck(dtype=tf.float32)

    # Decompress and transform the image back.
    z_shape = tf.concat([z_shape, [args.num_filters]], axis=0)
    z_hat = entropy_bottleneck.decompress(
        side_string, z_shape, channels=args.num_filters
    )
    sigma = hyper_synthesis_transform(z_hat)
    sigma = sigma[:, : y_shape[0], : y_shape[1], :]
    scale_table = np.exp(
        np.linspace(np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS)
    )
    conditional_bottleneck = RoundingGaussianConditional(
        sigma, scale_table, dtype=tf.float32
    )
    y_hat = conditional_bottleneck.decompress(string)
    x_hat = synthesis_transform(y_hat)

    # Remove batch dimension, and crop away any extraneous padding on the bottom
    # or right boundaries.
    x_hat = x_hat[0, : x_shape[0], : x_shape[1], :]

    # Write reconstructed image out as a PNG file.
    op = write_png(args.output_file, x_hat)

    # Load the latest model checkpoint, and perform the above actions.
    with tf.Session() as sess:
        latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
        tf.train.Saver().restore(sess, save_path=latest)
        sess.run(op, feed_dict=dict(zip(tensors, arrays)))


def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # High-level options.
    parser.add_argument(
        "--verbose",
        "-V",
        action="store_true",
        help="Report bitrate and distortion when training or compressing.",
    )
    parser.add_argument(
        "--num_filters", type=int, default=192, help="Number of filters per layer."
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="train",
        help="Directory where to save/load model checkpoints.",
    )
    parser.add_argument(
        "--qua_ent",
        choices={"AUN-Q", "STE-Q", "St-Q", "SRA-Q", "U-Q", "SGA-Q", "sigmoid", "DSl-Q", "DS-Q", "NonU-Q"},
        default="AUN-Q",
    )
    parser.add_argument(
        "--arch_param__k", default=24, type=int,
    )
    parser.add_argument(
        "--num_centers", default=6, type=int,
    )
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="What to do: 'train' loads training data and trains (or continues "
        "to train) a new model. 'compress' reads an image file (lossless "
        "PNG format) and writes a compressed binary file. 'decompress' "
        "reads a binary file and reconstructs the image (in PNG format). "
        "input and output filenames need to be provided for the latter "
        "two options. Invoke '<command> -h' for more information.",
    )

    # 'train' subcommand.
    train_cmd = subparsers.add_parser(
        "train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Trains (or continues to train) a new model.",
    )
    train_cmd.add_argument(
        "--train_root",
        help="path to the ImageNet train dir",
        default="/path/to/ImageNet/train/"
    )
    train_cmd.add_argument(
        "--train_file", default="../../datasets/ImageNet256.txt", help="paths of image files"
    )
    train_cmd.add_argument(
        "--batchsize", type=int, default=8, help="Batch size for training."
    )
    train_cmd.add_argument(
        "--patchsize", type=int, default=256, help="Size of image patches for training."
    )
    train_cmd.add_argument(
        "--lambda",
        type=float,
        default=0.01,
        dest="lmbda",
        help="Lambda for rate-distortion tradeoff.",
    )
    train_cmd.add_argument(
        "--last_step",
        type=int,
        default=1000000,
        help="Train up to this number of steps.",
    )
    train_cmd.add_argument(
        "--reduce_lr",
        action="store_true",
        default=False,
    )
    train_cmd.add_argument(
        "--adjust_aux_lr",
        action="store_true",
        default=False,
    )
    train_cmd.add_argument(
        "--mse_to_255",
        action="store_true",
        default=False,
    )
    train_cmd.add_argument(
        "--preprocess_threads",
        type=int,
        default=16,
        help="Number of CPU threads to use for parallel decoding of training "
        "images.",
    )
    train_cmd.add_argument(
        "--qua_dec",
        choices={"AUN-Q", "STE-Q", "St-Q", "SRA-Q", "U-Q", "SGA-Q", "SoftSTE", "sigmoid", "DSl-Q", "DS-Q", "Identity", "NonU-Q"},
        default="AUN-Q",
    )
    train_cmd.add_argument("--seed", type=int, default=0)
    # STH-Q
    train_cmd.add_argument("--fix_qua", action="store_true")
    # SRA-Q
    train_cmd.add_argument("--tau_decay_factor", type=float, default=0.0003)
    train_cmd.add_argument("--tau_decay_iteration", type=int, default=990000)
    # SGA-Q
    train_cmd.add_argument("--tau2_decay_factor", type=float, default=0.0003)
    train_cmd.add_argument("--tau2_decay_iteration", type=int, default=997000)
    # DS-Q, DSl-Q
    train_cmd.add_argument("--k_init", type=float, default=0.1)
    train_cmd.add_argument("--k_iteration", type=int, default=990000)
    train_cmd.add_argument("--k_end", type=float, default=0.1)
    # NonU-Q
    train_cmd.add_argument(
        "--regularization_factor_centers", default=0.1, type=float,
    )
    add_test_args(subparsers)

    # Parse arguments.
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args


def main(args):
    # Invoke subcommand.
    if args.command == "train":
        args.train_root = os.path.expanduser(args.train_root)
        train(args)
    elif args.command == "compress":
        if not args.output_file:
            args.output_file = args.input_file + ".tfci"
        compress(args)
    elif args.command == "decompress":
        if not args.output_file:
            args.output_file = args.input_file + ".png"
        decompress(args)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)

