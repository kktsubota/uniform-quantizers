import argparse
import random
import sys

from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

from common import add_test_args, prepare_dataset, read_png, write_png
from lib.decoder import Mentzer18Decoder, ThreeConvDecoder, FourConvDecoder, Cheng20Decoder
from lib.encoder import Mentzer18Encoder, ThreeConvEncoder, FourConvEncoder, Cheng20Encoder
from lib.ops import quantize, get_heatmap3D, mask_with_heatmap, transpose_NCHW_to_NHWC, transpose_NHWC_to_NCHW
from lib.quantizer import DSQ
from lib.entropy_model import RoundingEntropyBottleneck, Config, NonUQEntropyModel


def define_ae(name: str, num_filters: int, latent_size=None, heatmap: bool = False):
    if latent_size is None:
        latent_size = num_filters

    if heatmap:
        enc_latent_size = latent_size + 1
    else:
        enc_latent_size = latent_size

    if name == "balle17":
        analysis_transform = ThreeConvEncoder(num_filters, enc_latent_size)
        synthesis_transform = ThreeConvDecoder(num_filters)
    elif name == "balle18":
        analysis_transform = FourConvEncoder(num_filters, enc_latent_size)
        synthesis_transform = FourConvDecoder(num_filters)
    elif name == "mentzer18":
        analysis_transform = Mentzer18Encoder(num_filters, enc_latent_size)
        synthesis_transform = Mentzer18Decoder(num_filters)
    elif name == "cheng20":
        analysis_transform = Cheng20Encoder(num_filters, enc_latent_size)
        synthesis_transform = Cheng20Decoder(num_filters)
    else:
        raise NotImplementedError
    return analysis_transform, synthesis_transform


def train(args):
    """Trains the model."""
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    # Create input data pipeline.
    with tf.device("/cpu:0"):
        train_dataset = prepare_dataset(args)
    num_pixels = args.batchsize * args.patchsize ** 2

    # Get training patch from dataset.
    x = train_dataset.make_one_shot_iterator().get_next()

    # Instantiate model.
    analysis_transform, synthesis_transform = define_ae(args.autoencoder, args.num_filters, args.latent_size, args.heatmap)

    if args.qua_ent == "NonU-Q":
        pc_config = Config(args)
        entropy_bottleneck = NonUQEntropyModel(pc_config)
    else:
        entropy_bottleneck = RoundingEntropyBottleneck(approx=args.qua_ent, sub_mean=args.sub_ent_mean, stop_gradient=args.stop_gradient)

    # update hyper-parameters based on the current step
    step = tf.train.create_global_step()
    # - tau scheduler
    decaying_iter = tf.cast(step - args.tau_decay_iteration, tf.float32)
    # if decaying_iter < 0, tau should be 0.5.
    tau = tf.minimum(0.5, 0.5 * tf.exp(-args.tau_decay_factor * decaying_iter))
    # - tau2 scheduler (same with tau)
    decaying_iter2 = tf.cast(step - args.tau2_decay_iteration, tf.float32)
    # if decaying_iter < 0, tau should be 0.5.
    tau2 = tf.minimum(0.5, 0.5 * tf.exp(-args.tau2_decay_factor * decaying_iter2))
    # - T scheduling for sigmoid.
    # use iterations instead of epochs (1000k / 10k = 100)
    # https://github.com/aliyun/alibabacloud-quantization-networks/blob/master/anybit.py#L143
    T = (1.0 + tf.cast(step // args.T_iteration, tf.float32)) * args.T_factor
    # https://github.com/aliyun/alibabacloud-quantization-networks/blob/master/quan_weight_main.py#L202
    T = tf.cast(tf.minimum(T, 2000.0), tf.float32)
    # - k scheduling
    k_decay_iter = tf.cast(step - args.k_iteration, tf.float32)
    k = args.k_init + tf.maximum(0.0, (args.k_end - args.k_init) * k_decay_iter / (args.last_step - args.k_iteration))

    # replace hyper-parameters of entropy models
    if args.qua_ent == "SRA-Q":
        entropy_bottleneck.tau = tau
    elif args.qua_ent == "SGA-Q":
        entropy_bottleneck.tau2 = tau2
    elif args.qua_ent == "sigmoid":
        entropy_bottleneck.T = T
    elif args.qua_ent in {"DS-Q", "DSfix-Q"}:
        entropy_bottleneck.k = k

    # Build autoencoder.
    y = analysis_transform(x)

    if args.heatmap:
        y_nchw = transpose_NHWC_to_NCHW(y)
        heatmap3D = get_heatmap3D(y_nchw)
        y_masked = mask_with_heatmap(y_nchw, heatmap3D)
        y = transpose_NCHW_to_NHWC(y_masked)

    # obtain y_dec and y_tilde
    if args.fix_qua:
        # encoder is fixed
        # decoder can train with actual quantized value
        y_tilde, likelihoods = entropy_bottleneck(y, training=False)
        y_dec = y_tilde
    elif args.qua_dec == "SoftSTE":
        # Pan, et al., Three Gaps for Quantisation in Learned Image Compression, CVPRW 21
        # STE-Q followed by AUN-Q
        assert args.qua_ent == "AUN-Q"
        if args.sub_mean:
            # to define input spec
            entropy_bottleneck(y, training=True)
            _, _, _, input_slices = entropy_bottleneck._get_input_dims()
            medians = entropy_bottleneck._medians[input_slices]
        else:
            medians = None
        y_dec = quantize(y, medians, method="STE-Q")
        y_tilde, likelihoods = entropy_bottleneck(y_dec, training=True)

    else:
        y_tilde, likelihoods = entropy_bottleneck(y, training=True)

        # decoder quantization
        if args.qua_dec == args.qua_ent and args.qua_dec != "STE-Q":
            y_dec = y_tilde
        else:
            if args.sub_mean:
                _, _, _, input_slices = entropy_bottleneck._get_input_dims()
                medians = entropy_bottleneck._medians[input_slices]
            else:
                medians = None
            if args.qua_dec == "DS-Q":
                assert medians is None
                quantizer = DSQ(args.k_init)
                y_dec = quantizer(y)
            else:
                y_dec = quantize(y, medians, method=args.qua_dec, tau=tau, tau2=tau2, T=T, k=k)
            
    x_tilde = synthesis_transform(y_dec)

    # Total number of bits divided by number of pixels.
    if args.fix_ent:
        # to produce no gradient
        assert args.fix_qua
        train_bpp = 0
    else:
        # likelihoods -> bits
        if args.qua_ent != "NonU-Q":
            likelihoods = tf.log(likelihoods) / -np.log(2)

        H_real = tf.reduce_sum(likelihoods) / num_pixels
        if args.mask_loss:
            mask = transpose_NCHW_to_NHWC(heatmap3D)
            H_mask = tf.reduce_sum(likelihoods * mask) / num_pixels
            train_bpp = 0.5 * (H_mask + H_real)
        else:
            train_bpp = H_real

    if args.fix_dec:
        assert args.fix_qua
        train_mse = 0
    elif args.distortion == "msssim":
        train_mse = 1.0 - tf.reduce_mean(tf.image.ssim_multiscale(x, x_tilde, 1.0))
    elif args.distortion == "mse":
        # Mean squared error across pixels.
        train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
        # Multiply by 255^2 to correct for rescaling.
        train_mse *= 255 ** 2
    else:
        raise NotImplementedError

    # The rate-distortion cost.
    train_loss = args.lmbda * train_mse + train_bpp

    # add regularization of centers
    if args.qua_ent == "NonU-Q":
        reg = tf.to_float(pc_config.regularization_factor_centers)
        centers_reg = tf.identity(reg * tf.nn.l2_loss(entropy_bottleneck.centers), name='l2_reg')
        train_loss += centers_reg

    # Minimize loss and auxiliary loss, and execute update op.
    main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    if args.clip_grad is None:
        main_step = main_optimizer.minimize(train_loss, global_step=step)
    else:
        # https://stackoverflow.com/questions/36498127/how-to-apply-gradient-clipping-in-tensorflow
        grads_vars = main_optimizer.compute_gradients(train_loss)
        grads_clipped_vars = list()
        for grad, var in grads_vars:
            if grad is not None:
                grad = tf.clip_by_norm(grad, args.clip_grad)
            grads_clipped_vars.append((grad, var))
        main_step = main_optimizer.apply_gradients(grads_clipped_vars, global_step=step)
    
    if args.qua_ent != "NonU-Q":
        aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

    # ref: https://github.com/wiseodd/generative-models/blob/master/GAN/wasserstein_gan/wgan_tensorflow.py#L90
    clip_step = list()
    if args.qua_ent == "DS-Q":
        for w in entropy_bottleneck.quantizer.trainable_variables:
            op = w.assign(tf.clip_by_value(w, np.log(3), 1000))
            clip_step.append(op)
        tf.summary.scalar("k-dsq-ent", entropy_bottleneck.quantizer.k[0])

    elif args.qua_dec == "DS-Q":
        for w in quantizer.trainable_variables:
            op = w.assign(tf.clip_by_value(w, np.log(3), 1000))
            clip_step.append(op)
        tf.summary.scalar("k-dsq-dec", quantizer.k[0])

    if args.qua_ent != "NonU-Q":
        train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0], clip_step)
    else:
        train_op = tf.group(main_step, clip_step)

    tf.summary.scalar("loss", train_loss)
    tf.summary.scalar("bpp", train_bpp)
    tf.summary.scalar("mse", train_mse)
    tf.summary.scalar("tau", tau)
    tf.summary.scalar("tau2", tau2)
    tf.summary.scalar("T", T)
    tf.summary.scalar("k", k)
    tf.summary.scalar("step", step)
    if hasattr(entropy_bottleneck, "centers"):
        for i in range(args.num_centers):
            tf.summary.scalar(f"center_{i}", entropy_bottleneck.centers[i])

    # reduce event size
    # tf.summary.image("original", quantize_image(x))
    # tf.summary.image("reconstruction", quantize_image(x_tilde))

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

    analysis_transform, synthesis_transform = define_ae(args.autoencoder, args.num_filters, args.latent_size, args.heatmap)
    # Instantiate model.
    if args.qua_ent == "NonU-Q":
        pc_config = Config(args)
        entropy_bottleneck = NonUQEntropyModel(pc_config)
    else:
        entropy_bottleneck = RoundingEntropyBottleneck(approx=args.qua_ent)

    # Transform and compress the image.
    y = analysis_transform(x)

    if args.heatmap:
        y_nchw = transpose_NHWC_to_NCHW(y)
        heatmap3D = get_heatmap3D(y_nchw)
        y_masked = mask_with_heatmap(y_nchw, heatmap3D)
        y = transpose_NCHW_to_NHWC(y_masked)

    if args.qua_ent != "NonU-Q":
        string = entropy_bottleneck.compress(y)

    # Transform the quantized image back (if requested).
    y_hat, likelihoods = entropy_bottleneck(y, training=False)
    x_hat = synthesis_transform(y_hat)
    x_hat = x_hat[:, : x_shape[1], : x_shape[2], :]

    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

    # Total number of bits divided by number of pixels.
    if args.qua_ent == "NonU-Q":
        eval_bpp = tf.reduce_sum(likelihoods) / num_pixels
    else:
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
        if args.qua_ent != "NonU-Q":
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

            if args.qua_ent != "NonU-Q":
                # The actual bits per pixel including overhead.
                bpp = len(packed.string) * 8 / num_pixels
            else:
                bpp = 0.0
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
    entropy_bottleneck = RoundingEntropyBottleneck(approx=args.qua_ent)
    analysis_transform, synthesis_transform = define_ae(args.autoencoder, args.num_filters, args.latent_size, args.heatmap)

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
        "--autoencoder", default="balle17",
    )
    parser.add_argument(
        "--qua_ent",
        choices={"AUN-Q", "STE-Q", "St-Q", "SRA-Q", "U-Q", "SGA-Q", "sigmoid", "DS-Q", "DSfix-Q", "NonU-Q"},
        default="AUN-Q",
    )
    parser.add_argument(
        "--arch_param__k", default=24, type=int,
    )
    parser.add_argument(
        "--num_centers", default=6, type=int,
    )
    parser.add_argument(
        "--latent_size", default=None, type=int,
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="train",
        help="Directory where to save/load model checkpoints.",
    )
    parser.add_argument(
        "--heatmap", action="store_true", default=False
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
        "--preprocess_threads",
        type=int,
        default=16,
        help="Number of CPU threads to use for parallel decoding of training "
        "images.",
    )
    train_cmd.add_argument(
        "--qua_dec",
        choices={"AUN-Q", "STE-Q", "St-Q", "SRA-Q", "U-Q", "SGA-Q", "SoftSTE", "sigmoid", "DS-Q", "DSfix-Q", "Identity", "NonU-Q"},
        default="AUN-Q",
    )
    train_cmd.add_argument("--sub-mean", action="store_true", help="substract mean in dec.")
    train_cmd.add_argument("--sub-ent-mean", action="store_true", help="substract mean in ent.")
    train_cmd.add_argument("--seed", type=int, default=0)
    train_cmd.add_argument("--fix_ent", action="store_true")
    train_cmd.add_argument("--fix_dec", action="store_true")
    # STH-Q
    train_cmd.add_argument("--fix_qua", action="store_true")
    # SRA-Q
    train_cmd.add_argument("--tau_decay_factor", type=float, default=0.0003)
    train_cmd.add_argument("--tau_decay_iteration", type=int, default=990000)
    # SGA-Q
    train_cmd.add_argument("--tau2_decay_factor", type=float, default=0.0003)
    train_cmd.add_argument("--tau2_decay_iteration", type=int, default=997000)
    # sigmoid
    train_cmd.add_argument("--T_factor", type=float, default=1.0)
    train_cmd.add_argument("--T_iteration", type=int, default=10000)
    train_cmd.add_argument("--clip_grad", type=float, default=None, help="only used for sigmoid")
    # DS-Q, DSl-Q
    train_cmd.add_argument("--k_init", type=float, default=0.1)
    train_cmd.add_argument("--k_iteration", type=int, default=990000)
    train_cmd.add_argument("--k_end", type=float, default=0.1)
    # NonU-Q
    train_cmd.add_argument(
        "--regularization_factor_centers", default=0.1, type=float,
    )
    train_cmd.add_argument("--stop_gradient", action="store_true", default=False)
    train_cmd.add_argument("--sigma", type=float, default=1.0)
    train_cmd.add_argument("--distortion", default="mse", choices={"mse", "msssim"})
    train_cmd.add_argument("--mask-loss", action="store_true", default=False)

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
