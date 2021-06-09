import argparse
import os
import sys

from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc

from module import RoundingEntropyBottleneck


def read_png(filename):
    """Loads a PNG image file."""
    string = tf.read_file(filename)
    image = tf.image.decode_image(string, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255
    return image


def quantize_image(image):
    image = tf.round(image * 255)
    image = tf.saturate_cast(image, tf.uint8)
    return image


def write_png(filename, image):
    """Saves an image to a PNG file."""
    image = quantize_image(image)
    string = tf.image.encode_png(image)
    return tf.write_file(filename, string)


class AnalysisTransform(tf.keras.layers.Layer):
    """The analysis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(AnalysisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters,
                (9, 9),
                name="layer_0",
                corr=True,
                strides_down=4,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_0"),
            ),
            tfc.SignalConv2D(
                self.num_filters,
                (5, 5),
                name="layer_1",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_1"),
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
        super(AnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


class SynthesisTransform(tf.keras.layers.Layer):
    """The synthesis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(SynthesisTransform, self).__init__(*args, **kwargs)

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
        super(SynthesisTransform, self).build(input_shape)

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
        with open(args.train_file) as f:
            lines = f.readlines()
        train_files = [os.path.join(args.train_root, line.strip()) for line in lines]
        if not train_files:
            raise RuntimeError(
                "No training images found with glob '{}'.".format(args.train_glob)
            )
        train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
        train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
        train_dataset = train_dataset.map(
            read_png, num_parallel_calls=args.preprocess_threads
        )
        train_dataset = train_dataset.map(
            lambda x: tf.random_crop(x, (args.patchsize, args.patchsize, 3))
        )
        train_dataset = train_dataset.batch(args.batchsize)
        train_dataset = train_dataset.prefetch(32)

    num_pixels = args.batchsize * args.patchsize ** 2

    # Get training patch from dataset.
    x = train_dataset.make_one_shot_iterator().get_next()

    # Instantiate model.
    analysis_transform = AnalysisTransform(args.num_filters)
    if args.qua_ent == "noise":
        entropy_bottleneck = tfc.EntropyBottleneck()
    else:
        entropy_bottleneck = RoundingEntropyBottleneck(activation=args.qua_ent)
    synthesis_transform = SynthesisTransform(args.num_filters)

    # tau scheduler
    step = tf.train.create_global_step()
    decaying_iter = tf.cast(step - args.tau_decay_iteration, tf.float32)
    # if decaying_iter < 0, tau should be 0.5.
    tau = tf.minimum(0.5, 0.5 * tf.exp(-args.tau_decay_factor * decaying_iter))

    if args.qua_ent == "sga":
        entropy_bottleneck.tau = tau

    # Build autoencoder.
    y = analysis_transform(x)
    y_tilde, likelihoods = entropy_bottleneck(y, training=True)

    # decoder quantization
    if args.qua_dec == args.qua_ent:
        y_dec = y_tilde
    elif args.qua_dec == "noise":
        half = tf.constant(0.5)
        noise = tf.random.uniform(tf.shape(y), -half, half)
        y_dec = y + noise
    elif args.qua_dec == "deterministic":
        y_hat = tf.round(y)
        y_dec = tf.stop_gradient(y_hat - y) + y
    elif args.qua_dec in {"stochastic", "sga"}:
        diff = y - tf.floor(y)
        if args.qua_dec == "stochastic":
            probability = diff
        else:
            likelihood_up = tf.exp(-tf.atanh(diff) / tau)
            likelihood_down = tf.exp(-tf.atanh(1 - diff) / tau)
            probability = likelihood_down / (likelihood_up + likelihood_down)
        delta = tf.cast(
            (probability >= tf.random.uniform(tf.shape(probability))), tf.float32
        )
        y_hat = tf.floor(y) + delta
        y_dec = tf.stop_gradient(y_hat - y) + y
    elif args.qua_dec == "universal":
        # random value, shape: (N, 1, 1, 1)
        half = tf.constant(0.5)
        noise = tf.random.uniform(tf.shape(y), -half, half)[:, 0:1, 0:1, 0:1]
        y_univ = tf.round(y + noise) - noise
        y_dec = tf.stop_gradient(y_univ - y) + y
    else:
        raise NotImplementedError
    x_tilde = synthesis_transform(y_dec)

    # Total number of bits divided by number of pixels.
    train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

    # Mean squared error across pixels.
    train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
    # Multiply by 255^2 to correct for rescaling.
    train_mse *= 255 ** 2

    # The rate-distortion cost.
    train_loss = args.lmbda * train_mse + train_bpp

    # Minimize loss and auxiliary loss, and execute update op.
    main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    main_step = main_optimizer.minimize(train_loss, global_step=step)

    aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

    train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

    tf.summary.scalar("loss", train_loss)
    tf.summary.scalar("bpp", train_bpp)
    tf.summary.scalar("mse", train_mse)
    tf.summary.scalar("tau", tau)

    tf.summary.image("original", quantize_image(x))
    tf.summary.image("reconstruction", quantize_image(x_tilde))

    hooks = [
        tf.train.StopAtStepHook(last_step=args.last_step),
        tf.train.NanTensorHook(train_loss),
    ]
    with tf.train.MonitoredTrainingSession(
        hooks=hooks,
        checkpoint_dir=args.checkpoint_dir,
        save_checkpoint_secs=300,
        save_summaries_secs=60,
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
    analysis_transform = AnalysisTransform(args.num_filters)
    if args.qua_ent == "noise":
        entropy_bottleneck = tfc.EntropyBottleneck()
    else:
        entropy_bottleneck = RoundingEntropyBottleneck(activation=args.qua_ent)
    synthesis_transform = SynthesisTransform(args.num_filters)

    # Transform and compress the image.
    y = analysis_transform(x)
    string = entropy_bottleneck.compress(y)

    # Transform the quantized image back (if requested).
    y_hat, likelihoods = entropy_bottleneck(y, training=False)
    x_hat = synthesis_transform(y_hat)
    x_hat = x_hat[:, : x_shape[1], : x_shape[2], :]

    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

    # Total number of bits divided by number of pixels.
    eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

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
        tensors = [string, tf.shape(x)[1:-1], tf.shape(y)[1:-1]]
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
    x_shape = tf.placeholder(tf.int32, [2])
    y_shape = tf.placeholder(tf.int32, [2])
    with open(args.input_file, "rb") as f:
        packed = tfc.PackedTensors(f.read())
    tensors = [string, x_shape, y_shape]
    arrays = packed.unpack(tensors)

    # Instantiate model.
    if args.qua_ent == "noise":
        entropy_bottleneck = tfc.EntropyBottleneck()
    else:
        entropy_bottleneck = RoundingEntropyBottleneck(activation=args.qua_ent)
    synthesis_transform = SynthesisTransform(args.num_filters)

    # Decompress and transform the image back.
    y_shape = tf.concat([y_shape, [args.num_filters]], axis=0)
    y_hat = entropy_bottleneck.decompress(string, y_shape, channels=args.num_filters)
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
        "--num_filters", type=int, default=128, help="Number of filters per layer."
    )
    parser.add_argument(
        "--qua_ent",
        choices={"noise", "deterministic", "stochastic", "sga", "universal"},
        default="noise",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="train",
        help="Directory where to save/load model checkpoints.",
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
    )
    train_cmd.add_argument(
        "--train_file", default="datasets/ImageNetAll.txt", help="paths of image files"
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
        "--preprocess_threads",
        type=int,
        default=16,
        help="Number of CPU threads to use for parallel decoding of training "
        "images.",
    )
    train_cmd.add_argument(
        "--qua_dec",
        choices={"noise", "deterministic", "stochastic", "sga", "universal"},
        default="noise",
    )
    train_cmd.add_argument("--tau_decay_factor", type=float, default=0.0003)
    train_cmd.add_argument("--tau_decay_iteration", type=int, default=990000)

    # 'compress' subcommand.
    compress_cmd = subparsers.add_parser(
        "compress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a PNG file, compresses it, and writes a TFCI file.",
    )

    # 'decompress' subcommand.
    decompress_cmd = subparsers.add_parser(
        "decompress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a TFCI file, reconstructs the image, and writes back "
        "a PNG file.",
    )

    # Arguments for both 'compress' and 'decompress'.
    for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
        cmd.add_argument("input_file", help="Input filename.")
        cmd.add_argument(
            "output_file",
            nargs="?",
            help="Output filename (optional). If not provided, appends '{}' to "
            "the input filename.".format(ext),
        )

    # Parse arguments.
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args


def main(args):
    # Invoke subcommand.
    if args.command == "train":
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
